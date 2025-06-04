from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F

class Contriever:
    def __init__(self, all_questions):
        self.tokenizer = AutoTokenizer.from_pretrained("facebook/contriever")
        self.model = AutoModel.from_pretrained("facebook/contriever").eval().to("cuda" if torch.cuda.is_available() else "cpu")
        self.device = next(model.parameters()).device
        self.all_questions = all_questions
        self.question_embeddings = get_embedding(all_questions)

    def get_embedding(texts):
        inputs = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
            cls_embeddings = outputs.last_hidden_state[:, 0]
        return F.normalize(cls_embeddings, p=2, dim=1)

    def get_examples(question, k=5):
        question_embedding = get_embedding([question])
        similarities = torch.matmul(question_embedding, self.question_embeddings.T).squeeze(0)
        topk_indices = similarities.topk(k).indices.tolist()
        return [self.all_questions[i] for i in topk_indices]

    

