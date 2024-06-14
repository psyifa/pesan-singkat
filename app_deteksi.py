import pickle
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from PIL import Image
import base64
import pickle
import pandas as pd
from wordcloud import WordCloud

# Load saved model
with open('model_fraud.sav', 'rb') as model_file:
    deteksi_pesan = pickle.load(model_file)

# Load the pre-selected features for the vectorizer
with open('new_selected_feature_tf-idf.sav', 'rb') as vocab_file:
    selected_features = pickle.load(vocab_file)

# Initialize the TF-IDF Vectorizer with the loaded vocabulary set
loaded_vec = TfidfVectorizer(decode_error='replace', vocabulary=set(selected_features))
# Penting: Fit the vectorizer with an empty document to initialize it
loaded_vec.fit([""])

def set_background_image(image_path):
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode()
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/png;base64,{encoded_string}");
            background-size: cover;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Ganti 'image.png' dengan nama file gambar yang telah Anda unggah
set_background_image('bg4.png')

# Create a sidebar with navigation options
st.sidebar.title("Menu")
page = st.sidebar.radio("Pilih Halaman", ("Home", "Deteksi Pesan"))

if page == "Home":
    st.title('Selamat Datang di Aplikasi Deteksi Pesan Singkat')
    st.write("""
    Aplikasi ini membantu Anda mendeteksi pesan singkat (SMS) yang mungkin termasuk dalam kategori penipuan, spam, atau promosi. Dengan menggunakan model klasifikasi teks, aplikasi ini dapat mengidentifikasi jenis pesan yang Anda terima dan memberikan peringatan jika pesan tersebut berpotensi merugikan.
    """)
    # Displaying metrics with custom CSS
    col1, col2 = st.columns(2)
    
    # with col1:
    #     st.markdown('<div class="metric-box">', unsafe_allow_html=True)
    #     st.metric("Total Kerugian Akibat Fraud", "13.72", "triliun")
    #     st.markdown('</div>', unsafe_allow_html=True)

    # with col2:
    #     st.markdown('<div class="metric-box">', unsafe_allow_html=True)
    #     st.metric("Penerima Pesan Singkat Penipuan", "98.3 %", "Dari 1671 responden")
    #     st.markdown('</div>', unsafe_allow_html=True)

     # Displaying metrics
    col1, col2 = st.columns(2)
    col1.metric("Total Kerugian Akibat Fraud", "13.72", "triliun")
    col2.metric("Penerima Pesan Singkat Penipuan", "98.3 %", "Dari 1671 responden")

    # img = Image.open('sms700.png')
    # st.image(img, use_column_width=False)

        # Data for the visualization
    st.subheader("Modus Penipuan Digital dan Korbannya")
    modus_labels = [
        'Penipuan berkedok hadiah', 'Pengiriman tautan/link yang berisi malware/virus', 'Penipuan jual beli', 
        'Situs web/aplikasi palsu', 'Penipuan berkedok krisis keluarga', 'Pinjaman online ilegal', 
        'Penipuan berjenis amal atau bantuan sosial', 'Investasi bodong/palsu', 'Lowongan pekerjaan palsu', 
        'Pencurian identitas pribadi', 'Pembajakan/peretasan akun dompet digital', 'Penipuan arisan online', 
        'Penipuan berkedok asmara/romansa', 'Penerimaan sekolah/beasiswa palsu', 'Pemerasan pada proses penerimaan kerja'
    ]
    modus_sizes = [
        36.9, 33.8, 29.4, 27.4, 26.5, 21.9, 17.9, 17.4, 16.9, 16.4, 11.4, 11.1, 9.8, 8.5, 8.4
    ]

    # Create horizontal bar chart dengan latar belakang transparan
    fig, ax = plt.subplots(figsize=(10, 8))
    fig.patch.set_alpha(0.0)  # Set figure background to transparent
    ax.patch.set_alpha(0.0)  # Set axes background to transparent
    y_pos = np.arange(len(modus_labels))
    ax.barh(y_pos, modus_sizes, align='center', color='pink')  # Bar color to yellow
    ax.set_yticks(y_pos)
    ax.set_yticklabels(modus_labels, color='white')  # Set y-tick labels color to white
    ax.invert_yaxis()  # labels read top-to-bottom
    ax.set_xlabel('Persentase (%)', color='white')  # Set x-label color to white
    ax.set_title('Modus Penipuan Digital dan Korbannya', color='white')  # Set title color to white

    # Set the color of the ticks and spine
    ax.tick_params(axis='x', colors='white')  # Set x-axis tick color to white
    ax.tick_params(axis='y', colors='white')  # Set y-axis tick color to white
    ax.spines['top'].set_color('white')
    ax.spines['bottom'].set_color('white')
    ax.spines['left'].set_color('white')
    ax.spines['right'].set_color('white')

    # Display the chart
    st.pyplot(fig)



# Baca data dari file CSV
    df = pd.read_csv("clean_data.csv")

# Konversi kolom 'teks' ke string
    df['clean_teks'] = df['clean_teks'].astype(str)

# Filter data yang memiliki label 1 dan label 2
    df_label_1 = df[df['label'] == 1]
    df_label_2 = df[df['label'] == 2]

# Gabungkan semua teks menjadi satu string untuk label 1 dan label 2
    all_text_label_1 = " ".join(df_label_1['teks'])
    all_text_label_2 = " ".join(df_label_2['teks'])

# Create WordCloud with transparent background for label 1 and label 2
    wordcloud_label_1 = WordCloud(width=400, height=300, background_color="rgba(255, 255, 255, 0)", mode="RGBA").generate(all_text_label_1)
    wordcloud_label_2 = WordCloud(width=400, height=300, background_color="rgba(255, 255, 255, 0)", mode="RGBA").generate(all_text_label_2)

# Display WordClouds using Streamlit
    st.subheader("Kata yang Sering Muncul")

# Mengatur layout menjadi dua kolom
    col1, col2 = st.columns(2)

# Menampilkan WordCloud untuk Label 1 di kolom pertama
    with col1:
        st.write('<div style="display: flex; justify-content: center;"><h3 style="font-size: 16px;">Fraud</h3></div>', unsafe_allow_html=True)
        st.image(wordcloud_label_1.to_array())

# Menampilkan WordCloud untuk Label 2 di kolom kedua
    with col2:
        st.write('<div style="display: flex; justify-content: center;"><h3 style="font-size: 16px;">Promo</h3></div>', unsafe_allow_html=True)
        st.image(wordcloud_label_2.to_array())

#################################################################################################

elif page == "Deteksi Pesan":
    st.title('Identifikasi Penipuan Pesan Singkat')
    clean_teks = st.text_input('Masukkan Pesan')
    fraud_detection = ''

    detect_button = st.button('Hasil Deteksi')
    if detect_button:
        # Transform the input text using the loaded vectorizer
        transformed_text = loaded_vec.transform([clean_teks])
        predict_fraud = deteksi_pesan.predict(transformed_text)
        if predict_fraud[0] == 0:
            fraud_detection = 'SMS Normal'
            st.success(fraud_detection)
            st.write("Pesan ini tidak terindikasi sebagai penipuan atau promosi. Anda bisa membaca pesan ini dengan tenang.")
        elif predict_fraud[0] == 1:
            fraud_detection = 'SMS Fraud'
            st.success(fraud_detection)
            st.write("Pesan ini terindikasi sebagai penipuan. Jangan berikan informasi pribadi atau melakukan tindakan apapun yang diminta oleh pengirim pesan.")
        elif predict_fraud[0] == 2:
            fraud_detection = 'SMS Promo'
            st.success(fraud_detection)
            st.write("Pesan ini terindikasi sebagai promosi. Pastikan Anda berhati-hati dan memverifikasi tawaran yang diberikan sebelum mengambil tindakan lebih lanjut.")
        else:
            fraud_detection = 'SMS Spam'
            st.success(fraud_detection)
            st.write("Pesan ini terindikasi sebagai spam. Anda dapat mengabaikan atau menghapus pesan ini.")

    if not detect_button:
        st.info("""
         **Hint:**
        - **Fraud**: terdiri dari 3 kata atau lebih yang di dalamnya terkandung kalimat penipuan seperti "menang", "hadiah", "transfer", "akun diblokir", "klik tautan".
        - **Promo**: terdiri dari 5 kata atau lebih yang di dalamnya terkandung kalimat penawaran seperti "diskon", "promo", "gratis", "beli sekarang", "selamat", "dapatkan".
        """)