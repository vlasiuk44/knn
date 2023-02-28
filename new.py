import cv2
import numpy as np
import matplotlib.pyplot as plt
import pathlib
from tqdm import tqdm
from skimage.measure import regionprops, label, approximate_polygon, subdivide_polygon


def extract_features(image, i):
    if image.ndim == 3:
        gray = np.mean(image, 2)
        gray[gray > 0] = 1
        labeled = label(gray)
    else:
        labeled = image.astype("uint8")
    props = regionprops(labeled)[0]
    extants = props.extent
    eccentricity = props.eccentricity
    euler = props.euler_number
    rr, cc = props.centroid_local
    rr = rr / props.image.shape[0]
    cc = cc / props.image.shape[1]
    is_two_elements = i
    feret = (props.feret_diameter_max - 1) / np.max(props.image.shape)
    return np.array([extants, eccentricity, euler, rr, cc, feret, is_two_elements], dtype="f4") 


text_images = [plt.imread(path) for path in pathlib.Path(".").glob("*.png")]
train_images = {}

for path in tqdm(sorted(pathlib.Path("./train").glob("*"))):
    symbol = path.name[-1]
    train_images[symbol] = []
    for image_path in sorted(path.glob("*.png")):
        train_images[symbol].append(plt.imread(image_path))

train = []
responses = []
symb2class = {symbol: i for i, symbol in enumerate(train_images)}
class2symb = {value: key for key, value in symb2class.items()}
for i, symbol in tqdm(enumerate(train_images)):
    for image in train_images[symbol]:
        if symbol == "i":
            train.append(extract_features(image, True))
        else:
            train.append(extract_features(image, False))
        responses.append(symb2class[symbol])

train = np.array(train, dtype="f4")
responses = np.array(responses)

knn = cv2.ml.KNearest_create()
knn.train(train, cv2.ml.ROW_SAMPLE, responses)

for k in range(6):
    gray = cv2.cvtColor(text_images[k], cv2.COLOR_BGR2GRAY)
    gray[gray > 0] = 1
    labeled = label(gray)

    regions = regionprops(labeled)
    sortedLetters = sorted(
        regionprops(labeled),
        key=lambda r: r.bbox[1],
        reverse=False
    )

    spaces = []
    letter_i = []
    coord_i = []
    for i in range(len(sortedLetters)):
        if 0 < i < len(sortedLetters) - 1:
            if abs(sortedLetters[i].bbox[1] - sortedLetters[i + 1].bbox[1]) < 10 or abs(
                    sortedLetters[i].bbox[1] - sortedLetters[i - 1].bbox[1]) < 10:
                letter_i.append(sortedLetters[i].bbox)
                coord_i.append(i)
        if i < len(sortedLetters) - 1 and sortedLetters[i + 1].bbox[1] - sortedLetters[i].bbox[3] > 20:
            spaces.append(i)

    for i in range(0, len(letter_i), 2):
        lbn = labeled[letter_i[i + 1][0]:letter_i[i + 1][2], letter_i[i + 1][1]:letter_i[i + 1][3]]
        lbn[lbn > 0] = max(labeled[letter_i[i][0]:letter_i[i][2], letter_i[i][1]:letter_i[i][3]][3])
        labeled[letter_i[i + 1][0]:letter_i[i + 1][2], letter_i[i + 1][1]:letter_i[i + 1][3]] = lbn
        for j in range(len(coord_i)):
            if j >= i + 2:
                coord_i[j] -= 1
                spaces[j - 2] -= 1

    regions = regionprops(labeled)
    sortedLetters = sorted(
        regionprops(labeled),
        key=lambda r: r.bbox[1],
        reverse=False,
    )

    answer = []
    coord_i = coord_i[::2]
    for region in range(len(sortedLetters)):
        xbf = region

        if region in coord_i:
            features = extract_features(sortedLetters[xbf].image, True).reshape(1, -1)

        else:
            features = extract_features(sortedLetters[xbf].image, False).reshape(1, -1)
        ret, result, neighbours, dist = knn.findNearest(features, 2)
        "".join
        answer.append(class2symb[int(ret)])
    answer = "".join(answer)
    for j in spaces:
        answer = answer[:j + 1] + " " + answer[j + 1:]

    expected_vals = ['C is LOW-LEVEL', 'C++ is POWERFUL', 'Python is INTUITIVE', 'Rust is SAFE', 'LUA is EASY',
                     'Javascript is UGLY']
    min_len = min(len(answer), len(expected_vals[k]))
    mistakes = 0
    for j in range(min_len):
        if expected_vals[k][j] != answer[j]:
            mistakes += 1
    if len(answer) != len(expected_vals[k]):
        mistakes += abs(len(answer) - len(expected_vals[k]))
    print(f'Expected: {expected_vals[k]}, received: {answer}, error rate: {round(mistakes / len(expected_vals[k]), 2)}')
