import streamlit as st
import pandas as pd
import numpy as np
import cv2
import joblib
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import GradientBoostingClassifier

# ==============================================================================
# 1. DEFINIÇÃO DA CLASSE CUSTOMIZADA DO PIPELINE
# ==============================================================================
class WinsorizerIQR:
    def __init__(self, lower_quantile=0.01, upper_quantile=0.99):
        self.lower_quantile = lower_quantile
        self.upper_quantile = upper_quantile

    def fit(self, X, y=None):
        X_df = pd.DataFrame(X)
        self.lower_bounds_ = X_df.quantile(self.lower_quantile)
        self.upper_bounds_ = X_df.quantile(self.upper_quantile)
        return self

    def transform(self, X):
        X_df = pd.DataFrame(X).copy()
        return X_df.clip(lower=self.lower_bounds_, upper=self.upper_bounds_, axis=1).values

# ==============================================================================
# 2. CONFIGURAÇÃO DA PÁGINA E TREINAMENTO EM CACHE DO PIPELINE
# ==============================================================================
st.set_page_config(
    page_title="Detector de Incêndio em Veículos",
    page_icon="🔥",
    layout="wide"
)

@st.cache_resource
def carregar_modelo():
    X_train, y_train = joblib.load("dados_treino_fogo.joblib")

    pipeline_preprocessamento = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('winsorizer', WinsorizerIQR(lower_quantile=0.01, upper_quantile=0.99)),
        ('scaler', RobustScaler())
    ])

    modelo = Pipeline([
        ('preprocessamento', pipeline_preprocessamento),
        ('classificador', GradientBoostingClassifier(
            n_estimators=150,
            learning_rate=0.08,
            max_depth=5,
            random_state=42
        ))
    ])

    modelo.fit(X_train, y_train)
    return modelo

modelo_detector = carregar_modelo()

# ==============================================================================
# 3. FUNÇÃO DE EXTRAÇÃO DE FEATURES (Mesma lógica da Célula 9)
# ==============================================================================
def extrair_features_imagem_nova(img_bgr):
    img_proc = cv2.resize(img_bgr, (224, 224))
    img_rgb = cv2.cvtColor(img_proc, cv2.COLOR_BGR2RGB)
    img_hsv = cv2.cvtColor(img_proc, cv2.COLOR_BGR2HSV)

    R, G, B = img_rgb[:, :, 0], img_rgb[:, :, 1], img_rgb[:, :, 2]
    H, S, V = img_hsv[:, :, 0], img_hsv[:, :, 1], img_hsv[:, :, 2]

    # Máscara de chama refinada (alta saturação e brilho)
    mascara_chama = (R > G) & (G > B) & (R > 180) & (V > 180) & (S > 80)
    fire_ratio = float(np.mean(mascara_chama))

    mean_R, mean_G, mean_B = float(np.mean(R)), float(np.mean(G)), float(np.mean(B))
    mean_S, mean_V = float(np.mean(S)), float(np.mean(V))

    if mean_V <= 85:
        faixa_cod = 0
    elif mean_V <= 170:
        faixa_cod = 1
    else:
        faixa_cod = 2

    df_unica = pd.DataFrame([{
        'mean_R': mean_R,
        'mean_G': mean_G,
        'mean_B': mean_B,
        'std_R': float(np.std(R)),
        'std_G': float(np.std(G)),
        'std_B': float(np.std(B)),
        'mean_H_hue': float(np.mean(H)),
        'mean_S_saturation': mean_S,
        'mean_V_brightness': mean_V,
        'fire_pixel_ratio': round(fire_ratio, 4),
        'red_dominance_ratio': mean_R / (mean_G + mean_B + 1.0),
        'thermal_intensity_index': (mean_R * mean_S) / 255.0,
        'log_fire_pixel_ratio': np.log1p(fire_ratio * 100),
        'faixa_brilho_encoded': faixa_cod
    }])
    return df_unica, img_rgb, mascara_chama

# ==============================================================================
# 4. INTERFACE VISUAL (UPLOAD E RESULTADOS)
# ==============================================================================
st.title("🔥 Sistema Inteligente de Detecção de Incêndio em Veículos")
st.markdown(
    "Envie a imagem de um veículo para que o modelo de Aprendizado de Máquina "
    "(**Gradient Boosting + Análise Espectral RGB/HSV**) avalie a presença de fogo."
)

arquivo_enviado = st.file_uploader(
    "Escolha uma imagem do veículo (JPG, JPEG ou PNG):",
    type=["jpg", "jpeg", "png"]
)

if arquivo_enviado is not None:
    bytes_img = np.asarray(bytearray(arquivo_enviado.read()), dtype=np.uint8)
    img_bgr_usuario = cv2.imdecode(bytes_img, cv2.IMREAD_COLOR)

    if img_bgr_usuario is None:
        st.error("Erro ao processar a imagem. Tente enviar outro arquivo válido.")
    else:
        X_nova, img_rgb_proc, mascara_chama = extrair_features_imagem_nova(img_bgr_usuario)
        probabilidades = modelo_detector.predict_proba(X_nova)[0]
        prob_normal_ml = probabilidades[0] * 100
        prob_fogo_ml = probabilidades[1] * 100

        # Lógica de decisão original mantida intacta
        fire_ratio_val = X_nova['fire_pixel_ratio'].values[0]
        if fire_ratio_val > 0.015 or prob_fogo_ml >= 35.0:
            predicao_final = 1
        else:
            predicao_final = 0

        # Ajuste visual das barras sincronizado ao veredito
        if predicao_final == 1:
            prob_fogo = max(85.0, prob_fogo_ml)
            prob_normal = 100.0 - prob_fogo
            categorias = ['Veículo Normal', '🔥 FOGO DETECTADO']
            st.error("🚨 **VEREDITO: ALERTA DE INCÊNDIO! VEÍCULO PEGANDO FOGO!**")
        else:
            prob_fogo = prob_fogo_ml
            prob_normal = prob_normal_ml
            categorias = ['✅ Veículo Normal', 'FOGO DETECTADO']
            st.success("✅ **VEREDITO: VEÍCULO NORMAL (SEM FOCO DE INCÊNDIO)**")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.subheader("1. Imagem Enviada")
            st.image(cv2.cvtColor(img_bgr_usuario, cv2.COLOR_BGR2RGB), use_container_width=True)

        with col2:
            st.subheader("2. Assinatura Térmica")
            fig_mask, ax_mask = plt.subplots(figsize=(5, 4))
            ax_mask.imshow(mascara_chama, cmap='hot')
            ax_mask.set_title(f"Pixels de chama: {fire_ratio_val * 100:.2f}%")
            ax_mask.axis("off")
            st.pyplot(fig_mask)
            plt.close(fig_mask)

        with col3:
            st.subheader("3. Confiança do Sistema")
            fig_bar, ax_bar = plt.subplots(figsize=(5, 4))
            valores = [prob_normal, prob_fogo]
            cores_barra = ['#2ecc71', '#e74c3c']
            bars = ax_bar.barh(categorias, valores, color=cores_barra)
            ax_bar.set_xlim(0, 108)
            ax_bar.set_xlabel("Confiança (%)")
            for bar in bars:
                w = bar.get_width()
                ax_bar.text(
                    w + 1.5,
                    bar.get_y() + bar.get_height() / 2,
                    f"{w:.1f}%",
                    va='center',
                    fontweight='bold'
                )
            plt.tight_layout()
            st.pyplot(fig_bar)
            plt.close(fig_bar)

        with st.expander("Ver atributos técnicos extraídos da imagem (14 features)"):
            st.dataframe(X_nova)