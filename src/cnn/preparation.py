x_train = []

y_train = []

x_test = []

y_test = []


onlyfiles = [f for f in listdir(mypath+'aedes/') if isfile(join(mypath+'aedes/', f))]

X = []
Y = []

aux = [f for f in listdir(mypath+'culex/') if isfile(join(mypath+'culex/', f))]

onlyfiles = onlyfiles + aux
    
shuffle(onlyfiles)


for f in onlyfiles:
    print f
    if 'culex' in f:
        img = cv2.imread(mypath+'culex/{}'.format(f))
        Y.append(0)
    if 'aedes' in f:
        img = cv2.imread(mypath+'aedes/{}'.format(f))
        Y.append(1)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    (threshName, threshMethod) = ("THRESH_BINARY", cv2.THRESH_BINARY)
    (T, thresh) = cv2.threshold(gray, 150, 255, threshMethod)
    X.append(cv2.resize(thresh, (size_img, size_img)))
    
X = np.array(X)
Y = np.array(Y)    
    
X = X.reshape(X.shape[0], 1, size_img, size_img).astype('float32')
X = X/255

x_train = X[:size_train]

y_train = Y[:size_train]

x_test = X[size_train:]

y_test = Y[size_train:]

