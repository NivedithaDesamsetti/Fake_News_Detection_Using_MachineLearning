import pickle

# Load models
with open("models/all_models.pkl", "rb") as f:
    lr, nb, svm, vectorizer = pickle.load(f)

print("Models Loaded Successfully!")

news = input("Enter News Text:\n")

news_vector = vectorizer.transform([news])

# Predictions
lr_pred = lr.predict(news_vector)[0]
nb_pred = nb.predict(news_vector)[0]
svm_pred = svm.predict(news_vector)[0]

print("\n--- Predictions ---")
print("Logistic Regression:", lr_pred)
print("Naive Bayes:", nb_pred)
print("SVM:", svm_pred)

# Optional: Majority Voting
votes = [lr_pred, nb_pred, svm_pred]
final_prediction = max(set(votes), key=votes.count)

print("\nFinal Decision (Majority Voting):", final_prediction)