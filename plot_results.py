import matplotlib.pyplot as plt

rounds = [0, 1, 2, 3]
accuracy = [0.577, 0.902, 0.953, 0.962]
loss = [0.691, 0.226, 0.118, 0.122]

plt.figure(figsize=(10, 5))

# Plot Accuracy
plt.subplot(1, 2, 1)
plt.plot(rounds, accuracy, marker='o', color='green')
plt.title('Global Model Accuracy')
plt.xlabel('Round')
plt.ylabel('Accuracy')

# Plot Loss
plt.subplot(1, 2, 2)
plt.plot(rounds, loss, marker='o', color='red')
plt.title('Global Model Loss')
plt.xlabel('Round')
plt.ylabel('Loss')

plt.tight_layout()
plt.savefig('federated_performance.png')
plt.show()