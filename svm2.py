from sklearn import svm
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------
# STEP 1: Example data (30 students)
# Each row is: [hours studied, hours slept]
# ---------------------------------------------------

# Students who FAILED (label 0)
fail_students = [
    [1, 4], [2, 5], [1, 6], [3, 4], [2, 3],
    [0, 5], [1, 2], [3, 6], [4, 3], [2, 7],
    [3, 2], [4, 5], [1, 8], [0, 3], [4, 4],
]

# Students who PASSED (label 1)
pass_students = [
    [7, 8], [8, 7], [9, 9], [7, 6], [8, 8],
    [6, 7], [10, 8], [9, 6], [7, 9], [8, 5],
    [6, 6], [10, 10], [9, 7], [7, 5], [6, 8],
]

# Combine into one dataset
X = fail_students + pass_students
y = [0] * len(fail_students) + [1] * len(pass_students)

print("Total students:", len(X))

# ---------------------------------------------------
# STEP 2: Create and train the SVM
# ---------------------------------------------------
model = svm.SVC(kernel="linear")
model.fit(X, y)

print("Accuracy on training data:", model.score(X, y))
print("Number of support vectors:", len(model.support_vectors_))

# ---------------------------------------------------
# STEP 3: Predict for new students
# ---------------------------------------------------
new_students = [
    [2, 2],    # studied little, slept little
    [8, 9],    # studied a lot, slept a lot
    [5, 5],    # somewhere in the middle
    [1, 9],    # slept a lot but barely studied
    [9, 3],    # studied a lot but barely slept
]

predictions = model.predict(new_students)

for student, result in zip(new_students, predictions):
    label = "Pass" if result == 1 else "Fail"
    print(f"Student {student} -> {label}")

# ---------------------------------------------------
# STEP 4: Draw the result
# ---------------------------------------------------
X_arr = np.array(X)
new_arr = np.array(new_students)

# Training students (red = fail, blue = pass)
plt.scatter(X_arr[:, 0], X_arr[:, 1], c=y, cmap="coolwarm", s=80)

# Circle the support vectors (the points that define the line)
plt.scatter(model.support_vectors_[:, 0], model.support_vectors_[:, 1],
            s=220, facecolors="none", edgecolors="green", linewidths=2,
            label="Support vectors")

# New students as black stars
plt.scatter(new_arr[:, 0], new_arr[:, 1], c="black", marker="*", s=200,
            label="New students")

# Decision line: w1*x + w2*y + b = 0
w = model.coef_[0]
b = model.intercept_[0]
x_values = np.linspace(-1, 11)
y_values = -(w[0] * x_values + b) / w[1]
plt.plot(x_values, y_values, "k-", label="SVM boundary")

plt.xlim(-1, 11)
plt.ylim(0, 11)
plt.xlabel("Hours studied")
plt.ylabel("Hours slept")
plt.title("SVM: Pass vs Fail")
plt.legend()
plt.show()