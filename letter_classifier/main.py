import utils, random

print("Creating dataset...")
ENCODED_ALPHABET = utils.split_list(
    [ord(char) - ord("a") for char in "abcdefghijklmnopqrstuvwxyz"]
)

train_x: list[int] = []
train_y: list[int] = []

for i in range(len(ENCODED_ALPHABET)):
    for letter in ENCODED_ALPHABET[i]:
        train_x.append(letter)
        train_y.append(i)

combined = list(zip(train_x, train_y))
random.shuffle(combined)
train_x, train_y = zip(*combined)
del combined

print("Training...")
neuron = random.uniform(-1, 1)
bias = random.uniform(-1, 1)


def predict(value: float) -> float:
    return utils.sigmoid((neuron * value) + bias)


LEARNING_RATE = 0.01  # How fast the neuron learns, do not set it too high
EPOCHS = 1000  # How many times to iterate over the training data
for epoch in range(EPOCHS):
    epoch_loss = 0.0

    for i in range(len(train_x)):
        sample_x = train_x[i]
        sample_y = train_y[i]

        loss = sample_y - predict(sample_x)
        epoch_loss += loss

        neuron += sample_x * loss * LEARNING_RATE
        bias += loss

    print("Epoch", epoch + 1, "| Loss:", -epoch_loss if epoch_loss < 0 else epoch_loss)


print("Weight:", neuron, "| Bias:", bias)


while True:
    inp = input("Enter a character to predict (type exit to exit): ")

    if len(inp) > 1:
        if inp == "exit":
            break

        print("Invalid input!")
        continue

    prediction = predict(ord(inp) - ord("a"))

    print(
        "Predicted this letter to be from the",
        "first" if prediction < 0.5 else "second",
        f"half of the alphabet ({prediction})",
    )
