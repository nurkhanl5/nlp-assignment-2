import torch
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaTokenizer, RobertaModel
import torch.nn as nn
from torch.optim import AdamW

# Custom dataset class
class EntityDataset(Dataset):
    def __init__(self, inputs, main_labels, sub_labels, tokenizer, max_len=512):
        self.inputs = inputs
        self.main_labels = main_labels
        self.sub_labels = sub_labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        sample = self.inputs[index]
        main_label = self.main_labels[index]
        sub_label = self.sub_labels[index]

        # Tokenization
        tokenized = self.tokenizer(
            sample,
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt",
        )
        return {
            "input_ids": tokenized["input_ids"].squeeze(0),
            "attention_mask": tokenized["attention_mask"].squeeze(0),
            "sub_label": torch.tensor(sub_label, dtype=torch.float),
        }


class RoleClassifier(nn.Module):
    def __init__(self, num_main_classes, num_sub_classes, tokenizer_len):
        super(RoleClassifier, self).__init__()
        self.roberta = RobertaModel.from_pretrained("roberta-base")
        self.roberta.resize_token_embeddings(tokenizer_len)
        
        # Classification 
        self.main_head = nn.Linear(self.roberta.config.hidden_size, num_main_classes)
        self.sub_head = nn.Linear(self.roberta.config.hidden_size, num_sub_classes)
        
        self.softmax = nn.Softmax(dim=-1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_ids, attention_mask):
        output = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        pooled = output.pooler_output  # [CLS] representation

        # Predictions
        main_logits = self.main_head(pooled)
        sub_logits = self.sub_head(pooled)
        return self.softmax(main_logits), self.sigmoid(sub_logits)


def load_role_mappings(file_path):
    main_classes = {}
    sub_classes = {}
    current_mapping = None

    with open(file_path, "r") as file:
        for line in file:
            line = line.strip()
            if not line or line.startswith("#"): 
                continue
            
            if line.lower().startswith("main classes"):
                current_mapping = main_classes
                continue
            elif line.lower().startswith("subclasses"):
                current_mapping = sub_classes
                continue
            
            role, role_id = line.split(",")
            current_mapping[role.strip()] = int(role_id.strip())

    return main_classes, sub_classes


file_path = "role_mappings.txt"  
main_classes, sub_classes = load_role_mappings(file_path)

# Print to verify
print("Subclasses:", sub_classes)


# Read the data
data_path = "/kaggle/input/rawdocuments/raw-documents"
text_data, main_labels, sub_labels = [], [], []

with open(data_path, "r", encoding="utf-8") as file:
    for line in file:
        parts = line.strip().split("\t")
        if len(parts) < 5:
            continue
        filename, entity, start, end, main_class = parts[:5]
        subclasses = parts[5:]

        sub_vector = [0] * len(sub_classes)
        for subclass in subclasses:
            sub_vector[sub_classes[subclass]] = 1
        sub_labels.append(sub_vector)

        with open(f"EN/raw-documents/{filename}", "r", encoding="utf-8") as doc:
            text = doc.read()
            modified = (
                text[:int(start)]
                + "<entity> "
                + text[int(start):int(end)]
                + " </entity>"
                + text[int(end):]
            )
            text_data.append(modified)

# Tokenizer and dataset setup
tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
tokenizer.add_special_tokens({"additional_special_tokens": ["<entity>", "</entity>"]})

dataset = EntityDataset(
    inputs=text_data,
    main_labels=torch.tensor(main_labels),
    sub_labels=torch.tensor(sub_labels),
    tokenizer=tokenizer,
    max_len=512,
)

loader = DataLoader(dataset, batch_size=8, shuffle=True)

# Model and optimizer
model = RoleClassifier(num_main_classes=len(main_classes), num_sub_classes=len(sub_classes), tokenizer_len=len(tokenizer))
optimizer = AdamW(model.parameters(), lr=5e-5)

# Loss functions
sub_loss_fn = nn.BCELoss()

# Training
model.train()
for epoch in range(3):
    for batch in loader:
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]

        sub_label = batch["sub_label"]

        # Forward pass
        sub_preds = model(input_ids, attention_mask)
        loss = sub_loss_fn(sub_preds, sub_label)
     

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch + 1}, Loss: {loss.item()}")
    torch.save(model.state_dict(), f"roberta_epoch_{epoch + 1}.pth")
