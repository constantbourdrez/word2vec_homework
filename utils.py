import torch
import torch.nn as nn
import random
from torch.utils.data import DataLoader, Dataset
from tqdm.notebook import tqdm


def ce_loss(
        word_embedding: torch.Tensor,
        positive_context_embeddings: torch.Tensor,
        negative_context_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            word_embedding (torch.Tensor): Shape (batch_size, embedding_dim,)
            positive_context_embeddings (torch.Tensor): Shape (batch_size, 2R, embedding_dim)
            negative_context_embeddings (torch.Tensor): Shape (batch_size, 2KR, embedding_dim)
        Returns:
            torch.Tensor: The loss value
        """
        positive_similarity = torch.log(
            torch.sigmoid(
                torch.bmm(positive_context_embeddings, word_embedding.unsqueeze(-1) + 1e-10)
            )
        )
        negative_similarity = torch.log(
            torch.sigmoid(
                1
                - torch.bmm(negative_context_embeddings, word_embedding.unsqueeze(-1))
                + 1e-10
            )
        )
        return -torch.sum(positive_similarity) - torch.sum(negative_similarity)


class Word2Vec_Dataset(Dataset):
    def __init__(self, words, contexts):
        self.words = words
        self.contexts = contexts

    def __len__(self):
        return len(self.words)

    def __getitem__(self, idx):
        return torch.tensor(self.words[idx]), torch.tensor(self.contexts[idx])

class Word2Vec_Preprocessing():
    def __init__(self, tokenizer, R):
        self.tokenizer = tokenizer.from_pretrained("bert-base-uncased", do_lower_case=True)
        self. R = R

    def preprocessing_fn(self, x):
            x["review_token"] = self.tokenizer(
                x["review"],
                add_special_tokens=False,
                truncation=True,
                max_length=256,
                padding=False,
                return_attention_mask=False,
            )["input_ids"]
            x["label"] = 0 if x["sentiment"] == "negative" else 1
            return x

    def extract_words_contexts(self, ids_list):
        """
        Extract pairs of review_ids and their contexts from a list of review_ids.

        Args:
            ids_list (list): A list of review_ids.
            R (int): Radius for the context window (number of review_ids before and after each review).

        Returns:
            tuple: Two lists -
                1) List of review_ids.
                2) List of lists, where each inner list represents the context C+ of size 2R for each review_id.
        """
        words = []
        contexts = []

        for review in ids_list:
            for i in range(len(review)):
                w = review[i]

                # Extract the context window
                start = max(0, i - self.R)
                end = min(len(review), i + self.R + 1)

                # Create the context by taking elements before and after the current element
                context = review[start:i] + review[i + 1:end]

                # Pad the context to ensure it's exactly 2R in length
                left_padding = max(0, self.R - i)  # Amount of padding needed on the left
                right_padding = max(0, (i + self.R + 1) - len(review))  # Padding on the right

                context = [0] * left_padding + context + [0] * right_padding

                #context = [item for sublist in [context] for item in sublist]
                # Ensure final context length is exactly 2R
                context = context[:2 * self.R]

                words.append(w)
                contexts.append(context)

        return words, contexts
    def flatten_dataset_to_list(self, data):
        """
        Apply extract_words_contexts on the entire dataset and flatten the output into lists.

        Args:
            data (DataFrame): A pandas DataFrame with a column 'review_ids'.
            R (int): Radius for the context window (number of review_ids before and after each review).

        Returns:
            list: A flattened list of review_ids.
            list: A flattened list of their corresponding contexts (each context being a list).
        """

        words, contexts = [], []
        for i in tqdm(range(len(data)), desc="Processing dataset"):
            words_, contexts_ = self.extract_words_contexts([data["review_token"][i]])
            words.extend(words_)
            contexts.extend(contexts_)
        return words, contexts

    def collate_fn(self, batch, scaling_factor: int, vocab_size: int):
        batch_size = len(batch)
        review_token = torch.tensor([b[0] for b in batch])
        positive_context_ids = torch.tensor([b[1] for b in batch])
        positive_context_ids_set = set(positive_context_ids.flatten().tolist())
        negative_candidates = list(set(range(vocab_size)) - positive_context_ids_set)
        negative_context_ids = [
            random.sample(
                negative_candidates, scaling_factor * positive_context_ids.size(1)
            )
            for _ in range(batch_size)
        ]
        negative_context_ids = torch.tensor(negative_context_ids)
        result = {
            "review_token": review_token,
            "positive_context_ids": positive_context_ids,
            "negative_context_ids": negative_context_ids,
        }

    def preprocess(self, dataset, config, n_samples=5000):

        dataset = dataset.shuffle(seed=42)

        # Select 5000 samples
        dataset = dataset.select(range(n_samples))

        # Tokenize the dataset
        dataset = dataset.map(
            lambda x: self.preprocessing_fn(x),
            batched=False,
            num_proc=4,
        )
        # Remove useless columns
        dataset = dataset.remove_columns(["review", "sentiment"])

        # Split the train and validation
        train_size = int(0.8 * len(dataset))

        document_train_set = dataset.select(range(train_size))
        document_test_set = dataset.select(range(train_size, len(dataset)))

        print(len(dataset[3]["review_token"]))
        word, contexts = self.extract_words_contexts([dataset['review_token'][3]])
        print(len(contexts[0]), len(contexts[10]), len(contexts[-1]))


        train_dataset_words, train_dataset_contexts = self.flatten_dataset_to_list(
            document_train_set)

        test_dataset_words, test_dataset_contexts = self.flatten_dataset_to_list(
            document_test_set)

        train_dataset = Word2Vec_Dataset(train_dataset_words, train_dataset_contexts)
        test_dataset = Word2Vec_Dataset(test_dataset_words, test_dataset_contexts)

        train_dataloader = DataLoader(
                dataset=train_dataset,
                batch_size= config["Word2Vec_model"]["batch_size"],
                shuffle=True,
                collate_fn=lambda batch: self.collate_fn(
                    batch, scaling_factor=config["Word2Vec_model"]["ratio"], vocab_size= self.tokenizer.vocab_size
                ),
            )

        batch = next(iter(train_dataloader))
        print("batch ", batch)
        print("batch word_id size", batch["word_ids"].size())
        print("batch positive_context_ids size", batch["positive_context_ids"].size())
        print("batch negative_context_ids size", batch["negative_context_ids"].size())

        test_dataloader = DataLoader(
            test_dataset,
            batch_size=config["Word2Vec_model"]["batch_size"],
            shuffle=False,
            collate_fn=lambda batch: self.collate_fn(
                    batch, scaling_factor=config["Word2Vec_model"]["ratio"], vocab_size= self.tokenizer.vocab_size
                ),
            )

        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        return None



class Word2Vec_Model(nn.Module):
    def __init__(self, embedding_dim: int, vocab_size: int, device):
        super(Word2Vec_Model, self).__init__()
        self.device = device
        self.embbeding_dim = embedding_dim
        self.in_embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx= 0)
        self.out_embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx= 0)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.005)
        self.loss_fn = ce_loss


    def forward(self, target_word_ids, context_word_ids):
        embedded_target = self.in_embedding(target_word_ids)
        embedded_context = self.out_embedding(context_word_ids)

        # computation of the sigmoid of the dot product along the batch dimension (dim=2)
        score = torch.sigmoid(torch.sum(embedded_context*embedded_target, dim=2))
        return score

    def validation(self, valid_dataloader):

        # Tracking variables
        total_size = 0
        acc_total = 0
        loss_total = 0
        criterion = nn.BCELoss(reduction = 'none') # setting the reduction to none in order to not compute the mean outside the BCE Loss rather than inside

        # Set model to evaluation mode
        self.eval()

        # ========== Evaluation ==========

        with torch.no_grad():
            for batch in tqdm(valid_dataloader):

                # Pushing the batches to the computing device
                review_token = batch["review_token"].to(self.device)
                positive_context_ids = batch["positive_context_ids"].to(self.device)
                negative_context_ids = batch["negative_context_ids"].to(self.device)

                # Calculating the loss
                """ Note that the .unsqueeze(1) is used so that the score can be calculated. """
                pred_pos = self.forward(review_token.unsqueeze(1), positive_context_ids) # positive context prediction
                pred_neg = self.forward(review_token.unsqueeze(1), negative_context_ids) # negative context prediction
                """ For loss_positive and loss_negative, the mean of all BCE Losses for the positive/negative context predictions is computed. """
                loss_positive = torch.mean(criterion(pred_pos, torch.ones(pred_pos.shape, device=self.device)), dim=1)
                loss_negative = torch.mean(criterion(pred_neg, torch.zeros(pred_neg.shape, device=self.device)), dim=1)
                loss = torch.mean(loss_positive + loss_negative)
                loss_total += loss.detach().cpu().item()

                # Calculating the accuracy
                """ The threshold for the BCE Loss is set to 0.5. Under, the prediction is considered negative, and over positive """
                acc_positive = (pred_pos.squeeze() > 0.5)
                acc_negative = (pred_neg.squeeze() < 0.5)
                acc_total += acc_positive.int().sum().item()
                acc_total += acc_negative.int().sum().item() # summing the number of Trues
                total_size += acc_positive.numel()
                total_size += acc_negative.numel() # adding up all the predictions done

        # Set the model back to training mode
        self.train()

        return loss_total / len(valid_dataloader), acc_total / total_size

    def training(self, model, batch_size, n_epochs, train_dataloader, valid_dataloader):

        list_val_acc = []
        list_train_acc = []
        list_train_loss = []
        list_val_loss = []
        criterion = nn.BCELoss(reduction = 'none') # setting the reduction to none in order to not compute the mean outside the BCE Loss rather than inside

        for e in range(n_epochs):

            # ========== Training ==========

            # Set model to training mode
            self.train()

            # Tracking variables
            train_loss = 0
            epoch_train_acc = 0
            total_size = 0
            for batch in tqdm(train_dataloader):
                # Pushing the batches to the computing device
                review_token, positive_context_ids, negative_context_ids = (
                    batch["review_token"].to(self.device),
                    batch["positive_context_ids"].to(self.device),
                    batch["negative_context_ids"].to(self.device),
                )

                optimizer.zero_grad()
                #print(review_token.device, positive_context_ids.device, negative_context_ids.device)
                # Forward pass
                output_positive = self.forward(review_token.unsqueeze(1), positive_context_ids)
                output_negative = self.forward(review_token.unsqueeze(1), negative_context_ids)

                # Backward pass

                # Calculating the loss as in the validation function
                loss_positive = torch.mean(criterion(output_positive, torch.ones(output_positive.shape, device= self.device)), dim=1)
                loss_negative = torch.mean(criterion(output_negative, torch.zeros(output_negative.shape, device= self.device)), dim=1)
                loss = torch.mean(loss_positive + loss_negative)

                loss.backward()
                self.optimizer.step()
                train_loss += loss.detach().cpu().item()

                # Calculating the accuracy as in the validation function
                acc_positive = (output_positive.squeeze() > 0.5)
                acc_negative = (output_negative.squeeze() < 0.5)
                epoch_train_acc += acc_positive.int().sum().item()
                epoch_train_acc += acc_negative.int().sum().item()
                total_size += acc_positive.numel()
                total_size += acc_negative.numel()

            list_train_acc.append(epoch_train_acc / total_size)
            list_train_loss.append(train_loss / len(train_dataloader))

            # ========== Validation ==========

            l, a = validation(valid_dataloader)
            list_val_loss.append(l)
            list_val_acc.append(a)
            print(
                e,
                "\n\t - Train loss: {:.4f}".format(list_train_loss[-1]),
                "Train acc: {:.4f}".format(list_train_acc[-1]),
                "Val loss: {:.4f}".format(l),
                "Val acc:{:.4f}".format(a),
            )

        return list_train_loss, list_train_acc, list_val_loss, list_val_acc

    def save_model(self, path):
        torch.save(self.state_dict(), path)

    def load_model(self, path):
        self.load_state_dict(torch.load(path))


class Word2Vec_Wrapper():
    def __init__(self, device, tokenizer, dataset, config, n_samples=5000):
        self.device = device
        self.config = config
        self.R = config["Word2Vec_model"]["R"]
        self.word2vec_preprocess = Word2Vec_Preprocessing(tokenizer, self.R)
        self.word2vec_preprocess.preprocess(dataset, config, n_samples= n_samples)
        self.word2vec_model = Word2Vec_Model(embedding_dim=config["Word2Vec_model"]["embedding_dim"], vocab_size=self.word2vec_preprocess.tokenizer.vocab_size + 1, device= self.device)

    def get_dataloader(self):
        return self.word2vec_preprocess.train_dataloader, self.word2vec_preprocess.test_dataloader

    def train(self):
        self.word2vec_model.train(self.config["Word2Vec_model"]["num_epochs"], self.word2vec_preprocess.train_dataloader)

    def evaluate(self):
        self.word2vec_model.evaluate(self.word2vec_preprocess.tokenizer, self.preprocess.test_dataloader)

    def save_model(self, path):
        self.word2vec_model.save_model(path)
