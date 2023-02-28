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

train = []  #
responses = []  #
symb2class = {symbol: i for i, symbol in enumerate(train_images)}
class2symb = {value: key for key, value in symb2class.items()}
for i, symbol in tqdm(enumerate(train_images)):  #
    for image in train_images[symbol]:  #
        if symbol == "i":
            train.append(extract_features(image, True))
        else:
            train.append(extract_features(image, False))
        responses.append(symb2class[symbol])  #

train = np.array(train, dtype="f4")  #
responses = np.array(responses)

knn = cv2.ml.KNearest_create()  #
knn.train(train, cv2.ml.ROW_SAMPLE, responses)
expected_vals = ['C is LOW-LEVEL', 'C++ is POWERFUL', 'Python is INTUITIVE', 'Rust is SAFE', 'LUA is EASY',
                 'Javascript is UGLY']

for i in range(1):
    gray = cv2.cvtColor(text_images[5], cv2.COLOR_BGR2GRAY)
    gray[gray > 0] = 1
    labeled = label(gray)

    regions = regionprops(labeled)  # учитываю порядок букв
    sortedLetters = sorted(
        regionprops(labeled),
        key=lambda r: r.bbox[1],
        reverse=False,
    )

    spaces = []
    let_i = []
    coord_i = []
    for i in range(len(sortedLetters)):
        if 0 < i < len(sortedLetters) - 1:
            if abs(sortedLetters[i].bbox[1] - sortedLetters[i + 1].bbox[1]) < 10 or abs(
                    sortedLetters[i].bbox[1] - sortedLetters[i - 1].bbox[1]) < 10:
                let_i.append(sortedLetters[i].bbox)  # ищу i
                coord_i.append(i)
        # if i < len(sortedLetters) - 1 and sortedLetters[i + 1].bbox[1] - sortedLetters[i].bbox[3] > 20:
        #     spaces.append(i)  # ищу пробелы

    for i in range(0, len(let_i), 2):
        lbn = labeled[let_i[i + 1][0]:let_i[i + 1][2], let_i[i + 1][1]:let_i[i + 1][3]]
        lbn[lbn > 0] = max(labeled[let_i[i][0]:let_i[i][2], let_i[i][1]:let_i[i][3]][3])
        labeled[let_i[i + 1][0]:let_i[i + 1][2], let_i[i + 1][1]:let_i[i + 1][3]] = lbn
        for j in range(len(coord_i)):
            if j >= i + 2:
                coord_i[j] -= 1

    plt.imshow(labeled)
    plt.colorbar(label="Like/Dislike Ratio", orientation="horizontal")
    plt.show()

    regions = regionprops(labeled)
    sortedLetters = sorted(
        regionprops(labeled),
        key=lambda r: r.bbox[1],
        reverse=False,
    )
    # xbf=7
    # features = extract_features(sortedLetters[xbf].image).reshape(1, -1)
    # ret, result, neighbours, dist = knn.findNearest(features, 2)
    # print(class2symb[int(ret)])
    #
    # plt.imshow(sortedLetters[xbf].image)
    # plt.show()

    answer = []

    for region in range(len(sortedLetters)):

        # plt.imshow(sortedLetters[region].image)
        # plt.show()
        for j in range(0, len(coord_i), 2):
            if region == coord_i[j]:
                print(region)
                plt.imshow(sortedLetters[region].image)
                plt.show()
                features = extract_features(sortedLetters[region].image, True).reshape(1, -1)
            else:
                features = extract_features(sortedLetters[region].image, False).reshape(1, -1)
        ret, result, neighbours, dist = knn.findNearest(features, 2)
        answer.append(class2symb[int(ret)])

    answer = "".join(answer)
    # for i in spaces:
    #     answer = answer[:i + 1] + " " + answer[i + 1:]
    print(answer)
    print(spaces)

    print(coord_i)
