import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import os
import cv2
import glob
import squarify


st.set_page_config(
    page_title='Chess Pieces Computer Vision - EDA',
    layout='wide',
    initial_sidebar_state='expanded'
)

# function to help visualize each class in the dataset
def visualize_samples_by_label(df, label, num_samples=20):
    samples = df[df['label'] == label]['images'].iloc[:num_samples].tolist()
    num_cols = min(num_samples, 5)
    num_rows = (num_samples - 1) // num_cols + 1
    fig, axes = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(10, 2 * num_rows))
    count = 0
    for i in range(num_rows):
        for j in range(num_cols):
            if count < len(samples):
                sample = samples[count]
                img = cv2.imread(sample)
                ax = axes[i, j]
                ax.set_title(sample.split('/')[-1].split('\\')[-1])
                ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                ax.axis('off')
                count += 1
    plt.tight_layout()
    st.pyplot(fig)

# function to run the streamlit
def run():
    # Making the title
    st.title('Chess Pieces Computer Vision Model')

    # Making the Subheader
    st.subheader('Exploratory Data Analysis for the result of the chess pieces images')

    # Adding picture
    st.image('https://miro.medium.com/v2/resize:fit:1400/1*2Vif0hStfa-S8U5AkW_zJA.png',
             caption='Chess Checkmate - source from google')

    # Adding Description
    st.write('-'*50)
    st.write('Graded Challenge 7')
    st.write('Nama  : Achmad Abdillah Ghifari')
    st.write('Batch : BSD-006')
    st.write('-'*50)
    st.write('### OBJECTIVE')
    st.write('We want to create a computer vision model to classify chess pieces, turning a real chess game into a virtual one. The model will use computer vision techniques to recognize images based on certain characteristics. We will utilize a sequential model and also VGG19 model for improvement.')
    st.write('**(Please use the submenu on the left to navigate to the relevant feature that has been explored using exploratory data analysis)**')
    st.write('-'*50)

    # Creating the path to the dataset
    Chess_Data = r"C:\Users\THINKPAD X1 CAROBON\Hacktiv8\Phase 2\Week 1\Day 2\Deployment\Chessman-image-dataset\Chess"

    # Getting the image and making the dataframe
    bishop_files = glob.glob(os.path.join(Chess_Data, "Bishop", "*.jpg"))
    king_files = glob.glob(os.path.join(Chess_Data, "King", "*.jpg"))
    knight_files = glob.glob(os.path.join(Chess_Data, "Knight", "*.jpg"))
    pawn_files = glob.glob(os.path.join(Chess_Data, "Pawn", "*.jpg"))
    queen_files = glob.glob(os.path.join(Chess_Data, "Queen", "*.jpg"))
    rook_files = glob.glob(os.path.join(Chess_Data, "Rook", "*.jpg"))
    all_files = bishop_files + king_files + knight_files + pawn_files + queen_files + rook_files
    labels = ['bishop'] * len(bishop_files) + ['king'] * len(king_files) + ['knight'] * len(knight_files) + ['pawn'] * len(pawn_files) + ['queen'] * len(queen_files) + ['rook'] * len(rook_files)
    img_df = pd.DataFrame({'images': all_files, 'label': labels})
    img_df = img_df.sample(len(img_df)).reset_index(drop=True)
    class_names = ["Bishop", "King", "Knight", "Pawn", "Queen", "Rook"]
    image_counts = [len(os.listdir(os.path.join(Chess_Data, class_name))) for class_name in class_names]

    # Creating the sidebar for EDA
    submenu = st.sidebar.selectbox('Submenu', ['Dataset', 'Bishop', 'King', 'Knight', 'Pawn', 'Queen', 'Rook'])
    if submenu == "Dataset":
        st.write('## Dataset Information')
        with st.expander("Data Balance"):
            fig, ax = plt.subplots(figsize=(8, 5))
            colors = ["#252323", "#1B263B", "#462025", "#E7AF36", "#70798C", "#A39C8F"]
            squarify.plot(image_counts, label=class_names, ax=ax, text_kwargs={"fontsize": 10}, color=colors)
            plt.title("Chess Class Balance", fontsize=15)
            plt.axis("off")
            st.pyplot(fig)
        with st.expander("Total number of data"):
            st.write("Total number of images in the dataset:", sum(image_counts))
        with st.expander("Number of images in each class"):
            st.write("Number of images per class:", dict(zip(class_names, image_counts)))

    elif submenu == "Bishop":
        st.write('### Visualizing the Bishop class')
        visualize_samples_by_label(img_df, 'bishop', num_samples=20)
        with st.expander("Insight"):
            st.write('1. The bishop has a rounded top with a slit in the middle which looks like the hat of a bishop')
            st.write('2. The bishop is one of the tallest figures behind the king and queen piece')
            st.write('3. The bishop is considered a minor piece similar to the knight with a value of 3 pawns')
            st.write('4. The bishop can only move diagonally infinitely unless blocked by other pieces')

    elif submenu == "King":
        st.write('### Visualizing the King class')
        visualize_samples_by_label(img_df, 'king', num_samples=20)
        with st.expander("Insight"):
            st.write('1. The king has a cross on top of its head which symbolizes the crown the king wore')
            st.write('2. The King is the tallest piece in chess')
            st.write('3. The king is the most important piece in chess as losing this piece means losing the game')
            st.write('4. Despite the king importance its one of the weakest pieces behind a pawn as it can only move one square in any direction')
    
    elif submenu == "Knight":
        st.write('### Visualizing the Knight class')
        visualize_samples_by_label(img_df, 'knight', num_samples=20)
        with st.expander("Insight"):
            st.write('1. The knight has a unique shape as it resembles the head of a horse')
            st.write('2. The knight has an average height and size')
            st.write('3. The knight, similar to the bishop, is a minor piece with a value of 3 pawns')
            st.write('4. The knight is notorious for its unique movement of an L shape and being the only piece that can jump over pieces, which some argue makes it slightly more important than a bishop')
    
    elif submenu == "Pawn":
        st.write('### Visualizing the Pawn class')
        visualize_samples_by_label(img_df, 'pawn', num_samples=20)
        with st.expander("Insight"):
            st.write('1. The Pawns are usually designed as simple, cylindrical shapes with a flat top.')
            st.write('2. The Pawns is the smallest piece in the game')
            st.write('3. The pawn is the weakest piece in the game as it can only move one direction which is forward and only move one square except the beginning where it can move two and capture one piece diagonally')
            st.write('4. Despite the pawn being the weakest piece, it could be one of the strongest pieces due to how disposable and the threat they impose to higher-value pieces')
    
    elif submenu == "Queen":
        st.write('### Visualizing the Queen class')
        visualize_samples_by_label(img_df, 'queen', num_samples=20)
        with st.expander("Insight"):
            st.write('1. The queen has main features is a crown or regal headdress atop its head')
            st.write('2. The queen is the second highest piece in the game')
            st.write('3. The queen is the strongest piece in the game worth 9 pawns as it has the combined movement of a rook and a bishop thus having the most flexibility')
            st.write('4. Due to the strength of the queen, it is important to protect it like a king as losing a queen will be a major disadvantage for the player')

    elif submenu == "Rook":
        st.write('### Visualizing the Rook class')
        visualize_samples_by_label(img_df, 'rook', num_samples=20)
        with st.expander("Insight"):
            st.write('1. The rook is shaped like a tower with a crenellated top resembling a castle turret')
            st.write('2. The rook has an average height and size')
            st.write('3. The rook is a major piece worth 5 pawns')
            st.write('4. The rook can move as many squares vertically or horizontally as long as its not blocked')

if __name__ == '__main__':
    run()
