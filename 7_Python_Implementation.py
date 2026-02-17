import math
from collections import Counter

class NaiveBayesSpamFilter:
    def __init__(self, alpha=1):
        self.alpha = alpha
        self.spam_counts = Counter()
        self.ham_counts = Counter()
        self.spam_total_words = 0
        self.ham_total_words = 0
        self.vocab = set()
        self.p_spam = 0
        self.p_ham = 0

    def fit(self, train_data, labels):
        total_emails = len(labels)
        self.p_spam = sum(labels) / total_emails
        self.p_ham = 1 - self.p_spam

        for text, label in zip(train_data, labels):
            words = text.lower().split()
            for word in words:
                self.vocab.add(word)
                if label == 1:
                    self.spam_counts[word] += 1
                    self.spam_total_words += 1
                else:
                    self.ham_counts[word] += 1
                    self.ham_total_words += 1

    def predict(self, text):
        words = text.lower().split()
        v_size = len(self.vocab)

        spam_score = math.log(self.p_spam)
        ham_score = math.log(self.p_ham)

        for word in words:
            if word in self.vocab:
                p_word_spam = (self.spam_counts[word] + self.alpha) / (self.spam_total_words + self.alpha * v_size)
                p_word_ham = (self.ham_counts[word] + self.alpha) / (self.ham_total_words + self.alpha * v_size)

                spam_score += math.log(p_word_spam)
                ham_score += math.log(p_word_ham)

        return 1 if spam_score > ham_score else 0

train_text = ["Buy crypto now", "Buy gold now", "Buy milk please"]
labels = [1, 1, 0]

model = NaiveBayesSpamFilter()
model.fit(train_text, labels)

test_email = "crypto Buy"
result = model.predict(test_email)
print(f"The email '{test_email}' is classified as: {'Spam' if result == 1 else 'Ham'}")