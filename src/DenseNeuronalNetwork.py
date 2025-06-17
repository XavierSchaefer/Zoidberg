
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense, Input
# from tensorflow.keras.regularizers import l2
# from sklearn.metrics import recall_score
# from torchvision import transforms
# from imblearn.over_sampling import RandomOverSampler
# from src.dataset import Dataset
# from src.Model_Analyser import ModelAnalyser
# # 📦 Paramètres globaux
# height = 64     
# width = 64
# hiddenUnitsSize = 512
# hiddenUnitStacks = 5
# outputSize = 1
# generationSize = 20
# batchSize = 32

# def DenseNeuronalNetworkModel():
#     modelSize = height * width

#     print("🔄 Chargement des données...")
#     ImgTransform = transforms.Compose([
#         transforms.Resize((height, width)),
#         transforms.Grayscale(),
#         transforms.RandomRotation(10),
#         transforms.RandomHorizontalFlip(),
#         transforms.ToTensor()
#     ])
#     train_data = Dataset("dataset/initial/train", "img", ImgTransform)
#     x_train, y_train = Dataset.dataset_to_numpy(train_data.getDataset())
#     test_data = Dataset("dataset/initial/test", "img", ImgTransform)
#     x_test, y_test = Dataset.dataset_to_numpy(test_data.getDataset())
#     print(f"✅ {len(x_train)} images entraînement, {len(x_test)} images test")

#     # 🔁 Oversampling
#     print("🔁 Équilibrage du dataset avec RandomOverSampler...")
#     ros = RandomOverSampler(random_state=42)
#     x_train, y_train = ros.fit_resample(x_train, y_train)
#     print(f"✅ Dataset équilibré : {x_train.shape}")

#     # 🧠 Création du modèle
#     print("🧠 Création du modèle Dense (MLP)...")
#     model = Sequential()
#     model.add(Input(shape=(modelSize,)))  # ⬅️ Input à part


#     units = hiddenUnitsSize
#     for _ in range(hiddenUnitStacks):
#         model.add(Dense(units, activation='relu', kernel_regularizer=l2(0.001)))
#         units //= 2  # divise par 2 à chaque couche

#     model.add(Dense(outputSize, activation='sigmoid'))  # 🔽 Sortie sigmoid pour classification

#     print("⚙️ Compilation du modèle...")
#     model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

#     print("🚀 Entraînement...")
#     history = model.fit(
#         x_train,
#         y_train, 
#         epochs=generationSize, 
#         batch_size=batchSize, 
#         validation_split=0.2,
#         class_weight={0:5,1:1},
#         )


#     print("🧪 Évaluation finale sur le jeu de test...")
#     test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
#     print(f"\n📉 Test Loss : {test_loss:.4f}")
#     print(f"✅ Test Accuracy : {test_acc:.4f}")
   


#     # 🔎 Prédictions
#     y_pred_probs = model.predict(x_test)
#     y_pred = (y_pred_probs > 0.6).astype(int)  # 0 ou 1

#     recall = recall_score(y_test, y_pred)
#     print(f"📌 Recall global : {recall:.4f}")
#     model_config = {
#         "height": height,
#         "width": width,
#         "hiddenUnitsSize": hiddenUnitsSize,
#         "hiddenUnitStacks": hiddenUnitStacks,
#         "generationSize": generationSize,
#         "batchSize": batchSize,
#         "test_acc": test_acc,
#         "test_loss": test_loss,
#     }
#     analyser = ModelAnalyser(history, y_test, y_pred, model_config, save_dir="resultsDenseNeuronalNetwork")
#     analyser.plot()

#     return model