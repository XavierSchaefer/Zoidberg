
# from torchvision import transforms
# from src.dataset import Dataset
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import classification_report
# from imblearn.over_sampling import RandomOverSampler

# def randomForestModel():
#     ImgTransform = transforms.Compose([
#         transforms.Resize((96, 96)),
#         transforms.Grayscale(),
#         transforms.ToTensor()
#     ]) 

#     # Chargement et flatten
#     test_data = Dataset("dataset/initial/test", "img", ImgTransform)
#     train_data = Dataset("dataset/initial/train", "img", ImgTransform)

#     x_train, y_train = Dataset.dataset_to_numpy(train_data.getDataset())
#     x_test, y_test = Dataset.dataset_to_numpy(test_data.getDataset())

#     # ░░░ Sous-échantillonnage (balanced MIN)
#     x_train_balmin, y_train_balmin = Dataset.balance_dataset(x_train, y_train)
#     model_min = RandomForestClassifier(n_estimators=1000, class_weight="balanced", n_jobs=-1, random_state=42)
#     model_min.fit(x_train_balmin, y_train_balmin)
#     y_pred_min = model_min.predict(x_test)

#     # ░░░ Sur-échantillonnage (RandomOverSampler)
#     ros = RandomOverSampler(random_state=42)
#     x_train_ros, y_train_ros = ros.fit_resample(x_train, y_train)
#     model_ros = RandomForestClassifier(n_estimators=1000, class_weight="balanced", n_jobs=-1, random_state=42)
#     model_ros.fit(x_train_ros, y_train_ros)
#     y_pred_ros = model_ros.predict(x_test)

#     # ░░░ Affichage des résultats
#     print("📉 Rapport (Sous-échantillonnage)")
#     print(classification_report(y_test, y_pred_min, target_names=train_data.getDataset().classes))

#     print("📈 Rapport (Sur-échantillonnage)")
#     print(classification_report(y_test, y_pred_ros, target_names=train_data.getDataset().classes))
