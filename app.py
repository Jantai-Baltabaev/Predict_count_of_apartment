import base64
import os
import requests
from io import BytesIO

import gradio as gr
import hdbscan
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def load_model_from_secret(secret_name):
    url = os.environ.get(secret_name)
    if url is None:
        raise ValueError(f"Секрет {secret_name} не найден!")
    resp = requests.get(url)
    resp.raise_for_status()
    return joblib.load(BytesIO(resp.content))

# Загружаем все модели и вспомогательные объекты
catboost_model = load_model_from_secret("CATBOOST_MODEL")
cat_lower     = load_model_from_secret("CAT_LOWER")
cat_upper     = load_model_from_secret("CAT_UPPER")

rf_pipeline   = load_model_from_secret("RF_PIPELINE")

sgd_model     = load_model_from_secret("SGD_MODEL")
sgd_bagging   = load_model_from_secret("SGD_BAGGING")

hdbscan_model = load_model_from_secret("HDBSCAN_MODEL")
cat_options   = load_model_from_secret("CAT_OPTIONS")

knn_model     = load_model_from_secret("KNN_MODEL")
scaler_knn    = load_model_from_secret("SCALER_KNN")
knn_columns   = load_model_from_secret("KNN_COLUMNS")
y_train       = load_model_from_secret("Y_TRAIN")


def predict_price(room_count, lat, lon, series, material,
                  floor, total_floors, total_area,
                  heating, condition):
    # 1. Валидация
    errs = []
    for name, val, lo, hi in [
        ("Количество комнат", room_count, 1, 20),
        ("Площадь (м²)", total_area, 1, 1500),
        ("Этаж", floor, 0, 40),
        ("Общее число этажей", total_floors, 1, 40)
    ]:
        if val is None:
            errs.append(f"<li>Укажите {name}.</li>")
        elif not (lo <= val <= hi):
            errs.append(f"<li>{name} должно быть от {lo} до {hi}.</li>")

    if lat is None or lon is None or not (42.800 <= lat <= 42.950 and 74.500 <= lon <= 74.750):
        errs.append("<li>Координаты должны быть внутри Бишкека.</li>")
    if floor is not None and total_floors is not None and floor > total_floors:
        errs.append("<li>Этаж не может превышать общее число этажей.</li>")

    if errs:
        return f"""
        <div class="error-container">
          <div class="error-icon">⚠️</div>
          <div class="error-text-content">
            <h4 class="error-title">Пожалуйста, исправьте:</h4>
            <ul class="error-list">{"".join(errs)}</ul>
          </div>
        </div>
        """

    # 2. Кластер по HDBSCAN
    lbl, _ = hdbscan.approximate_predict(hdbscan_model, np.array([[lat, lon]]))
    cluster = str(int(lbl[0]))

    # 3. Подготовка DataFrame
    cols = ['room_count','lat','lon','Серия','house_material','floor',
            'total_floors','total_area','Отопление','Состояние','hdbscan_cluster']
    df = pd.DataFrame([[room_count, lat, lon, series, material,
                        floor, total_floors, total_area, heating,
                        condition, cluster]], columns=cols)

    # 4. CatBoost предсказания
    p_cat   = catboost_model.predict(df)[0]
    low_cat = cat_lower.predict(df)[0]
    hi_cat  = cat_upper.predict(df)[0]

    # 5. RandomForest предсказания
    rf_model        = rf_pipeline.named_steps['rf']
    X_rf            = rf_pipeline.named_steps['preprocess'].transform(df)
    preds_trees     = [t.predict(X_rf)[0] for t in rf_model.estimators_]
    p_rf            = np.mean(preds_trees)
    low_rf, hi_rf   = np.percentile(preds_trees, [2.5, 97.5])

    # 6. SGD+Bagging предсказания
    p_sgd           = sgd_model.predict(df)[0]
    preds_sgd       = [m.predict(df)[0] for m in sgd_bagging.estimators_]
    low_sgd, hi_sgd = np.percentile(preds_sgd, [2.5, 97.5])

    # 7. Среднее по трем моделям
    avg_pred = np.mean([p_cat, p_rf, p_sgd])

    # 8. KNN-анализ соседей
    dfn = df.copy()
    num_cols = dfn.select_dtypes(float).columns.tolist()
    dfn[num_cols] = scaler_knn.transform(dfn[num_cols])
    dfn = pd.get_dummies(dfn).reindex(columns=knn_columns, fill_value=0)

    k = 30 if p_cat<=100_000 else 20 if p_cat<=250_000 else 10 if p_cat<=400_000 else 5
    idxs = knn_model.kneighbors(dfn, n_neighbors=k, return_distance=False)[0]
    neigh = y_train.iloc[idxs]
    m_knn      = neigh.mean()
    low_knn, hi_knn = np.percentile(neigh, [2.5, 97.5])

    # 9. Строим гистограмму соседей
    fig, ax = plt.subplots(figsize=(5,3), dpi=120)
    ax.hist(neigh, bins=10, alpha=0.6)
    ax.axvline(avg_pred, color='red',   label='Avg 3 models')
    ax.axvline(m_knn,    color='green', linestyle='--', label='KNN mean')
    ax.legend(fontsize=9)
    buf = BytesIO()
    plt.savefig(buf, format="png", bbox_inches='tight')
    buf.seek(0)
    img64 = base64.b64encode(buf.read()).decode()
    plt.close(fig)

    # Помощная функция форматирования
    fmt = lambda x: f"{int(x):,}".replace(',', ' ')

    # 10. Сборка HTML-вывода
    return f"""
    <div class="output-container">
      <h2>💰 Средняя цена: {fmt(avg_pred)} $</h2>
      <div class="cards">
        <div><b>CatBoost:</b> {fmt(p_cat)} $<br><small>95%: {fmt(low_cat)}–{fmt(hi_cat)}</small></div>
        <div><b>RF:</b> {fmt(p_rf)} $<br><small>95%: {fmt(low_rf)}–{fmt(hi_rf)}</small></div>
        <div><b>SGD:</b> {fmt(p_sgd)} $<br><small>95%: {fmt(low_sgd)}–{fmt(hi_sgd)}</small></div>
        <div><b>KNN mean:</b> {fmt(m_knn)} $<br><small>95%: {fmt(low_knn)}–{fmt(hi_knn)}</small></div>
      </div>
      <img src="data:image/png;base64,{img64}" alt="KNN Histogram"/>
    </div>
    """

# 11. CSS-оформление
custom_css = """
.output-container { text-align:center; font-family:Arial, sans-serif; }
.cards { display:flex; justify-content: space-around; margin:10px 0; }
.cards div { background:#f5f5f5; padding:10px; border-radius:5px; width:22%; }
"""

# 12. Собираем Gradio Blocks
with gr.Blocks(css=custom_css, title="Оценка квартир в Бишкеке") as demo:
    gr.Markdown("# 🏠 Predict Bishkek Apartment Price")
    with gr.Row():
        with gr.Column():
            room    = gr.Number(label="Комнат",      value=2)
            area    = gr.Number(label="Площадь (м²)", value=75)
            floor   = gr.Number(label="Этаж",        value=3)
            totfl   = gr.Number(label="Всего этажей",value=9)
            lat     = gr.Number(label="Широта",      value=42.875)
            lon     = gr.Number(label="Долгота",     value=74.603)
            ser     = gr.Dropdown(sorted(cat_options["Серия"]),      label="Серия")
            mat     = gr.Dropdown(cat_options["house_material"],     label="Материал")
            heat    = gr.Dropdown(cat_options["Отопление"],         label="Отопление")
            cond    = gr.Dropdown(cat_options["Состояние"],         label="Состояние")
            btn     = gr.Button("Рассчитать")
        with gr.Column():
            out     = gr.HTML()

    btn.click(predict_price,
              inputs=[room, lat, lon, ser, mat, floor, totfl, area, heat, cond],
              outputs=out)

if __name__ == "__main__":
    demo.launch()
