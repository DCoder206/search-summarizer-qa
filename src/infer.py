from transformers import pipeline, BartTokenizer, AutoTokenizer, AutoModelForQuestionAnswering, AutoModelForSeq2SeqLM, AutoModelForSequenceClassification, AutoModel
import requests, re, nltk, torch, numpy as np
from bs4 import BeautifulSoup
import urllib.parse
from sklearn.metrics.pairwise import cosine_similarity
from os.path import exists
from random import sample as random_sample
from spacy import load as spc_load

class SessionManager:
    def __init__(self, max_tokens=1024):
        self.tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")
        self.max_tokens = max_tokens
        self.context = ""
    def update_context(self, new_content):
        combined = f"{self.context}\n{new_content}".strip()
        tokens = self.tokenizer.tokenize(combined)
        if len(tokens) > self.max_tokens:
            keep_tokens = int(self.max_tokens * 0.35)
            new_tokens = self.tokenizer.tokenize(new_content)
            tokens = tokens[:keep_tokens] + new_tokens[-(self.max_tokens - keep_tokens):]
        self.context = self.tokenizer.convert_tokens_to_string(tokens)
    def get_context(self):
        return self.context.strip()

class REINFORCETrainer:
    def __init__(self, model, tokenizer, learning_rate=4e-5):
        self.model = model
        self.tokenizer = tokenizer
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    def update(self, inputs, responses, rewards):
        rewards_tensor = torch.tensor(rewards, dtype=torch.float32).to(self.model.device)
        if isinstance(self.model, AutoModelForQuestionAnswering):
            outputs = self.model(**inputs)
            start_logits = outputs.start_logits
            end_logits = outputs.end_logits
            start_probs = torch.nn.functional.log_softmax(start_logits, dim=-1)
            end_probs = torch.nn.functional.log_softmax(end_logits, dim=-1)
            start_positions = responses["start_positions"]
            end_positions = responses["end_positions"]
            start_log_probs = start_probs.gather(-1, start_positions.unsqueeze(-1)).squeeze(-1)
            end_log_probs = end_probs.gather(-1, end_positions.unsqueeze(-1)).squeeze(-1)
            total_log_probs = (start_log_probs + end_log_probs) / 2
            loss = -torch.mean(total_log_probs * rewards_tensor)
        elif isinstance(self.model, AutoModelForSequenceClassification):
            outputs = self.model(**inputs)
            log_probs = torch.nn.functional.log_softmax(outputs.logits, dim=-1)
            response_labels = responses["labels"]
            selected_log_probs = log_probs.gather(-1, response_labels.unsqueeze(-1)).squeeze(-1)
            loss = -torch.mean(selected_log_probs * rewards_tensor)
        else:
            outputs = self.model(**inputs, labels=responses["input_ids"])
            log_probs = torch.nn.functional.log_softmax(outputs.logits, dim=-1)
            response_ids = responses["input_ids"]
            response_mask = (response_ids != self.tokenizer.pad_token_id).float()
            nll = torch.nn.functional.nll_loss(log_probs.view(-1, log_probs.size(-1)), response_ids.view(-1), reduction='none')
            log_probs = -nll.view(response_ids.size()) * response_mask
            avg_log_probs = (log_probs.sum(dim=-1) / response_mask.sum(dim=-1)).mean()
            loss = -avg_log_probs * rewards_tensor.mean()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

class WebQASystem:
    def __init__(self, query, model_names=None, rl_reward_factors=None, max_tokens=1024):
        self.query = query.lower()
        self.session = SessionManager(max_tokens)
        self.rl_reward_factors = rl_reward_factors if rl_reward_factors is not None else {"qa": 0.5, "relevance": 0.3, "summarizer": 0.2}
        self.model_paths = {"summarizer": "../models/fine_tuned_bart", "qa": "../models/fine_tuned_qa", "relevance": "../models/fine_tuned_relevance"}
        self.model_defaults = {"summarizer": "facebook/bart-base", "qa": "distilbert-base-cased-distilled-squad", "relevance": "cross-encoder/nli-roberta-base"}
        if model_names is not None:
            self.model_paths.update(model_names)
        self.max_tokens = max_tokens
        self.content_dict = {}
        self.page_summaries = {}
        self.search_results = []
        self.combined_summaries = ""
        self.sumry = ""
        self.retrieved_context = ""
        self.qa_result = None
        self.full_query = f"{self.session.get_context()}\n\nCurrent Query: {self.query}"
        self.classifier = pipeline("zero-shot-classification", model="cross-encoder/nli-roberta-base")
        self.tokenizer_sum = BartTokenizer.from_pretrained("facebook/bart-base")
        self.summarizer = pipeline("summarization", model="facebook/bart-base")
        self.tokenizer_emb = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
        self.model_emb = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2").eval()
        self.tokenizer_rk = AutoTokenizer.from_pretrained("cross-encoder/ms-marco-MiniLM-L-6-v2")
        self.model_rk = AutoModelForSequenceClassification.from_pretrained("cross-encoder/ms-marco-MiniLM-L-6-v2")
        self.qa_pipeline = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")
        self.nlp = spc_load("en_core_web_md")
        self.init_rlhf_models()
    def init_rlhf_models(self):
        self.model_summarizer, self.tokenizer_summarizer = self.init_model("summarizer", AutoModelForSeq2SeqLM)
        self.model_qa, self.tokenizer_qa = self.init_model("qa", AutoModelForQuestionAnswering)
        self.model_relevance, self.tokenizer_relevance = self.init_model("relevance", AutoModelForSequenceClassification)
        self.trainers = {
            "summarizer": REINFORCETrainer(self.model_summarizer, self.tokenizer_summarizer),
            "qa": REINFORCETrainer(self.model_qa, self.tokenizer_qa),
            "relevance": REINFORCETrainer(self.model_relevance, self.tokenizer_relevance)
        }
    def init_model(self, key, model_class):
        if exists(self.model_paths[key]):
            model = model_class.from_pretrained(self.model_paths[key])
            tokenizer = AutoTokenizer.from_pretrained(self.model_paths[key])
        else:
            model = model_class.from_pretrained(self.model_defaults[key])
            tokenizer = AutoTokenizer.from_pretrained(self.model_defaults[key])
        return model, tokenizer
    def extract_text(self, html):
        text = re.sub(r'<[^>]*>', ' ', html)
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[?:;\'"(){}[\]<>|@&\/\\]', '', text)
        text = re.sub(r'%[^%]*%', '', text)
        text = re.sub(r'%', '', text)
        words = text.split()
        filtered_words = [word for word in words if not (len(word) > 45 or (word.isupper() and len(word) > 32))]
        return " ".join(filtered_words).strip()
    def filter_relevant_sentences(self, text, query, threshold=0.7):
        sentences = nltk.sent_tokenize(text)
        candidate_labels = ["entailment", "contradiction"]
        hypothesis_template = f"This sentence {{}} is relevant to the query: {query}"
        valid_sentences = [s for s in sentences if s.strip()]
        if not valid_sentences:
            return ""
        results = self.classifier(valid_sentences, candidate_labels, hypothesis_template=hypothesis_template, batch_size=4)
        filtered_sentences = []
        for sentence, result in zip(valid_sentences, results):
            score = result["scores"][result["labels"].index("entailment")]
            if score >= threshold:
                filtered_sentences.append(sentence)
        return " ".join(filtered_sentences)
    def chunk_text(self, text, max_tokens=512):
        sentences = nltk.sent_tokenize(text, language="english")
        full_text_token_count = len(self.tokenizer_sum.tokenize(text))
        if full_text_token_count <= max_tokens:
            return [text]
        chunks = []
        current_chunk = []
        current_length = 0
        for sentence in sentences:
            sentence_tokens = self.tokenizer_sum.tokenize(sentence)
            sentence_token_count = len(sentence_tokens)
            if current_length + sentence_token_count <= max_tokens:
                current_chunk.append(sentence)
                current_length += sentence_token_count
            else:
                if current_chunk:
                    chunks.append(" ".join(current_chunk))
                    current_chunk = [sentence]
                    current_length = sentence_token_count
                else:
                    chunks.append(sentence[:512])
                    current_chunk = []
                    current_length = 0
        if current_chunk:
            chunks.append(" ".join(current_chunk))
        return chunks
    def summarize_large_text(self, text, query, max_chunk_tokens=512, threshold=0.7):
        filtered_text = self.filter_relevant_sentences(text, query, threshold)
        if not filtered_text.strip():
            return ""
        chunks = self.chunk_text(filtered_text, max_chunk_tokens)
        summaries = []
        for chunk in chunks:
            if not chunk.strip():
                continue
            try:
                input_tokens = self.tokenizer_sum.tokenize(chunk)
                input_length = len(input_tokens)
                adjusted_max_length = min(200, input_length) if input_length > 10 else 10
                adjusted_min_length = min(50, adjusted_max_length - 1) if adjusted_max_length > 1 else 0
                summary = self.summarizer(chunk, max_length=adjusted_max_length, min_length=adjusted_min_length, do_sample=False, truncation=True)[0]["summary_text"]
                summaries.append(summary)
            except Exception:
                continue
        if not summaries:
            return ""
        final_summary = " ".join(summaries)
        while len(self.tokenizer_sum.tokenize(final_summary)) > 512:
            input_tokens = self.tokenizer_sum.tokenize(final_summary)
            input_length = len(input_tokens)
            adjusted_max_length = min(512, input_length)
            adjusted_min_length = min(100, adjusted_max_length - 1) if adjusted_max_length > 1 else 0
            final_summary = self.summarizer(final_summary, max_length=adjusted_max_length, min_length=adjusted_min_length, do_sample=False, truncation=True)[0]["summary_text"]
        return final_summary
    def search_and_extract(self):
        search_url = f"https://html.duckduckgo.com/html/?q={'+'.join(self.query.split(' '))}"
        response = requests.get(search_url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:135.0) Gecko/20100101 Firefox/135.0"})
        soup = BeautifulSoup(response.text, "html.parser")
        results = []
        for i, result in enumerate(soup.find_all("div", class_="result"), start=1):
            if i > 6:
                break
            title = result.find("a", class_="result__a")
            if not title:
                continue
            tag = result.find("a", class_="result__snippet")
            snip = tag.text.strip() if tag else None
            results.append({"id": i, "url": title["href"], "description": snip})
        self.search_results = results
        for result in results:
            try:
                url = result["url"]
                mch = re.search(r"uddg=([^&]+)", url)
                if mch:
                    url = urllib.parse.unquote(mch.group(1))
                page_response = requests.get(url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:135.0) Gecko/20100101 Firefox/135.0"})
                soup_page = BeautifulSoup(page_response.text, "html.parser")
                for element in soup_page(["script", "style", "nav", "footer", "header"]):
                    element.decompose()
                page_text = self.extract_text(page_response.text)[:7500]
                if page_text:
                    self.content_dict[result["id"]] = page_text
            except Exception:
                continue
    def summarize_pages(self):
        for doc_id, text in self.content_dict.items():
            page_prompt = f"Extract information relevant to '{self.query}' from this text:\n\n{text}"
            page_summary = self.summarize_large_text(page_prompt, self.query)
            self.page_summaries[doc_id] = page_summary
        self.combined_summaries = " ".join(self.page_summaries.values())
        final_prompt = f"Considering previous context: {self.session.get_context()}\nSynthesize information relevant to '{self.query}' from these summaries:\n\n{self.combined_summaries}"
        self.sumry = self.summarize_large_text(final_prompt, self.full_query)
    def get_embedding(self, text):
        inputs = self.tokenizer_emb(text, return_tensors="pt", truncation=True, max_length=256)
        with torch.no_grad():
            outputs = self.model_emb(**inputs)
        embeddings = outputs.last_hidden_state
        attention_mask = inputs["attention_mask"]
        masked_embeddings = embeddings * attention_mask.unsqueeze(-1).float()
        mean_embedding = masked_embeddings.sum(dim=1) / attention_mask.sum(dim=1, keepdim=True).clamp(min=1e-9)
        return mean_embedding[0].numpy()
    def rerank(self, query, contexts, top_k=4):
        scores = []
        for context in contexts:
            inputs = self.tokenizer_rk(query, context, return_tensors="pt", truncation=True)
            with torch.no_grad():
                scores.append(self.model_rk(**inputs).logits.item())
        return [c for _, c in sorted(zip(scores, contexts), reverse=True)[:top_k]]
    def augmented_retrieval(self):
        rag_chunks = self.chunk_text(" ".join(self.content_dict.values()), 256)
        chunk_embeds = np.array([self.get_embedding(chunk) for chunk in rag_chunks])
        query_embed = self.get_embedding(self.query).reshape(1, -1)
        similarity_scores = cosine_similarity(query_embed, chunk_embeds)[0]
        top_k = 4
        ini_indices = np.argsort(similarity_scores)[:top_k]
        ini_context = [rag_chunks[i] for i in ini_indices]
        self.retrieved_context = " ".join(self.rerank(self.query, ini_context, top_k=4))
    def perform_qa(self):
        query_context = f"{self.sumry}\n\n{self.retrieved_context}"
        self.qa_result = self.qa_pipeline(question=self.query, context=query_context)
    def update_session(self):
        self.session.update_context(f"query: {self.query}\nanswer: {self.qa_result['answer']}\nsummary context: {self.sumry}\nretrieved Context: {self.retrieved_context}")
    def get_feedback(self):
        try:
            feedback = int(input("Rate the answer quality (1-5):"))
            return max(1, min(5, feedback))
        except ValueError:
            return 3
    def rlhf_update(self):
        user_rating = self.get_feedback()
        user_rating = max(1, min(5, user_rating))
        base_reward = (user_rating - 3) / 2
        rewards = {k: base_reward * v for k, v in self.rl_reward_factors.items()}
        rag_chunks = self.chunk_text(" ".join(self.content_dict.values()), 256)
        query_embed = self.get_embedding(self.query).reshape(1, -1)
        chunk_embeds = np.array([self.get_embedding(chunk) for chunk in rag_chunks])
        similarity_scores = cosine_similarity(query_embed, chunk_embeds)[0]
        top_k = 4
        ini_indices = np.argsort(similarity_scores)[:top_k]
        negative_indices = np.argsort(similarity_scores)[:top_k]
        positive_chunks = [rag_chunks[i] for i in ini_indices]
        negative_chunks = [rag_chunks[i] for i in negative_indices]
        context_pairs = [(self.query, chunk, 1) for chunk in positive_chunks] + [(self.query, chunk, 0) for chunk in negative_chunks]
        np.random.shuffle(context_pairs)
        relevance_inputs = self.tokenizer_relevance.batch_encode_plus([(self.session.get_context(), c) for _, c, _ in context_pairs], padding=True, truncation=True, return_tensors="pt")
        relevance_labels = torch.tensor([l for _, _, l in context_pairs]).long()
        with torch.no_grad():
            logits = self.model_relevance(**relevance_inputs).logits
            preds = torch.sigmoid(logits[:, 0]).round().squeeze()
            labels = torch.tensor([l for _, _, l in context_pairs]).float()
            accuracy = (preds == labels).float().mean()
        rewards["relevance"] = accuracy.item() * rewards["relevance"]
        all_qa_chunks = self.chunk_text(" ".join(self.content_dict.values()), 256)
        query_context = f"{self.sumry}\n\n{self.retrieved_context}"
        query_context_chunks = self.chunk_text(query_context, 256)
        negative_qa_chunks = [c for c in all_qa_chunks if c not in query_context_chunks]
        adversarial_qa_chunks = random_sample(negative_qa_chunks, min(2, len(negative_qa_chunks)))
        augmented_qa_context = f"{query_context}[ADV]{'[ADV]'.join(adversarial_qa_chunks)}"
        answer_embed = self.get_embedding(self.qa_result["answer"])
        context_embed = self.get_embedding(augmented_qa_context)
        consistency_score = cosine_similarity([answer_embed], [context_embed])[0][0]
        qa_reward = 0.5 * base_reward + 0.3 * consistency_score + 0.2 * cosine_similarity([self.get_embedding(self.qa_result["answer"])], [self.get_embedding(self.session.get_context())])[0][0]
        doc_source = self.nlp(self.combined_summaries)
        doc_summary = self.nlp(self.sumry)
        source_ents = set((ent.text.lower(), ent.label_) for ent in doc_source.ents)
        summary_ents = set((ent.text.lower(), ent.label_) for ent in doc_summary.ents)
        entity_retention = len(summary_ents & source_ents) / len(source_ents) if source_ents else 1.0
        compression_ratio = len(self.sumry) / len(self.combined_summaries)
        compression_score = max(0, 1 - abs(0.45 - compression_ratio))
        summarizer_reward = 0.3 * base_reward + 0.4 * entity_retention + 0.2 * compression_score + 0.1 * cosine_similarity([self.get_embedding(self.sumry)], [self.get_embedding(self.session.get_context())])[0][0]
        inputs = (
            {"name": "summarizer", "query": self.tokenizer_summarizer(self.sumry, return_tensors="pt", truncation=True), "response": self.tokenizer_summarizer(self.sumry, return_tensors="pt"), "reward": [summarizer_reward * self.rl_reward_factors["summarizer"]]},
            {"name": "qa", "query": self.tokenizer_qa(self.query, augmented_qa_context, return_tensors="pt", truncation=True, padding=True), "response": {"start_positions": torch.tensor([self.qa_result["start"]]), "end_positions": torch.tensor([self.qa_result["end"]])}, "reward": [qa_reward * self.rl_reward_factors["qa"]]},
            {"name": "relevance", "query": self.tokenizer_relevance.batch_encode_plus([(self.session.get_context(), c) for _, c, _ in context_pairs], padding=True, truncation=True, return_tensors="pt"), "response": relevance_labels, "reward": [rewards["relevance"] * self.rl_reward_factors["relevance"]]}
        )
        for input_dict in inputs:
            trainer = self.trainers[input_dict["name"]]
            trainer.update(inputs=input_dict["query"], responses=input_dict["response"], rewards=input_dict["reward"])
        for model_name in self.model_paths:
            self.trainers[model_name].model.save_pretrained(self.model_paths[model_name])
            self.trainers[model_name].tokenizer.save_pretrained(self.model_paths[model_name])
    def run(self):
        self.search_and_extract()
        self.summarize_pages()
        self.augmented_retrieval()
        self.perform_qa()
        print(self.qa_result["answer"])
        self.update_session()
        self.rlhf_update()
if __name__ == "__main__":
    user_query = input("Enter your query: ")
    rlhf_system = WebQASystem(user_query)
    rlhf_system.run()
