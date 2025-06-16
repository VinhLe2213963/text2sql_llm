import torch
import torch.nn.functional as F

class Retriever:
    def __init__(self, all_questions, question_embeddings, tokenizer, model, client=None):
        self.tokenizer = tokenizer
        self.model = model
        self.client = client
        self.device = next(model.parameters()).device if type(model) is not str else "cpu"
        self.all_questions = all_questions
        self.question_embeddings = question_embeddings

    def get_embedding(self, texts):
        if "gemini" in self.model:
            result = self.client.models.embed_content(
                model=self.model.split("/")[1],
                contents=texts)
            return torch.tensor([item.values for item in result.embeddings])

        inputs = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
            cls_embeddings = outputs.last_hidden_state[:, 0]
        return F.normalize(cls_embeddings, p=2, dim=1)

    def get_examples(self, question_embedding, k=5):
        # question_embeddings = self.get_embedding(self.all_questions)
        # question_embedding = self.get_embedding([question])
        similarities = torch.matmul(question_embedding, self.question_embeddings.T).squeeze(0)
        topk_indices = similarities.topk(k).indices.tolist()
        return [self.all_questions[i] for i in topk_indices]