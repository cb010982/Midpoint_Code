import streamlit as st
import pandas as pd
import torch
import torch.nn as nn

# Load user data
file_path = "filtered_df.csv"
df = pd.read_csv(file_path)

# Create a mapping for user indexing within model limits
if "user_index" not in df.columns:
    unique_users = df["user_id"].unique()
    user_mapping = {uid: idx for idx, uid in enumerate(unique_users)}
    df["user_index"] = df["user_id"].map(user_mapping)

print(df[["user_id", "user_index"]].drop_duplicates())

# Load meal metadata
course_df = pd.read_csv("course_cleaned.csv")
course_df["category"] = course_df["category"].str.lower()

# Define the NeuralFM model 
class NeuralFM(nn.Module):
    def __init__(self, num_users, num_items, embed_dim, deep_layers):
        super(NeuralFM, self).__init__()
        self.user_embedding = nn.Embedding(num_users, embed_dim)
        self.item_embedding = nn.Embedding(num_items, embed_dim)
        self.fm_layer = nn.Linear(embed_dim * 2, 1)
        layers = []
        input_dim = embed_dim  
        for layer_size in deep_layers:
            layers.append(nn.Linear(input_dim, layer_size))
            layers.append(nn.ReLU())
            input_dim = layer_size
        layers.append(nn.Linear(input_dim, 1)) 
        self.dnn = nn.Sequential(*layers)
        self.sigmoid = nn.Sigmoid()

    def forward(self, user, item):
        user_embed = self.user_embedding(user)
        item_embed = self.item_embedding(item)
        fm_input = torch.cat([user_embed, item_embed], dim=1)
        interaction = user_embed * item_embed
        fm_output = self.fm_layer(fm_input)
        deep_output = self.dnn(interaction)
        return self.sigmoid(fm_output + deep_output).squeeze()

# Hyperparameters for the model
num_users = 1575  
num_items = 5751 
embed_dim = 32  
deep_layers = [128, 64] 

# Initialize the model and load saved components
model = NeuralFM(num_users, num_items, embed_dim, deep_layers)
model.fm_layer.load_state_dict(torch.load("nfm_fm_layer.pth"))
model.dnn.load_state_dict(torch.load("nfm_dnn.pth"))
model.user_embedding.load_state_dict(torch.load("nfm_user_embedding.pth"))
model.item_embedding.load_state_dict(torch.load("nfm_item_embedding.pth"))
model.eval()
print("Model successfully loaded in Streamlit!")

# Define sugar thresholds and classification function
sugar_thresholds = {
    "high": {"max_sugar": 15, "max_calories": 350, "min_fiber": 2},
    "normal": {"max_sugar": 30, "max_calories": 500, "min_fiber": 1.5},
    "low": {"max_sugar": 40, "max_calories": 700, "min_fiber": 1}
}

def classify_sugar_level(sugar_value):
    if sugar_value > 180:
        return "high"
    elif 100 <= sugar_value <= 180:
        return "normal"
    else:
        return "low"

# Streamlit UI
st.title("🍽️ Personalized Meal Recommendations")

# if "user_id" in st.session_state:
#     if st.button("Sign Out"):
#         del st.session_state["user_id"]
#         if "sugar_value" in st.session_state:
#             del st.session_state["sugar_value"]
#         st.experimental_rerun()

# User Authentication
choice = st.radio("Choose an option:", ["Sign In", "Sign Up"])

def ask_sugar_level():
    return st.number_input("Enter your sugar level (mg/dL):", min_value=80.0, max_value=300.0, step=0.1)

if choice == "Sign In":
    user_name = st.text_input("Enter your name:")
    if st.button("Sign In"):
        if user_name in df["Name"].values:
            user_id = df.loc[df["Name"] == user_name, "user_index"].values[0]  
            st.session_state["user_id"] = user_id
            st.success(f"Welcome back, {user_name}! Your User ID is {user_id}.")
            st.session_state["sugar_value"] = ask_sugar_level()
        else:
            st.error("User not found. Please sign up.")

elif choice == "Sign Up":
    user_name = st.text_input("Enter a new username:")
    if st.button("Sign Up"):
        if user_name in df["Name"].values:
            st.error("This username already exists. Try signing in.")
        else:
            if len(df) < num_users:
                new_user_index = df["user_index"].max() + 1 if len(df) > 0 else 0 
                new_user = pd.DataFrame({"user_id": [new_user_index], "user_index": [new_user_index], "Name": [user_name]})
                df = pd.concat([df, new_user], ignore_index=True)
                df.to_csv(file_path, index=False)
                df = pd.read_csv(file_path)  
                st.session_state["user_id"] = new_user_index
                st.success(f"User {user_name} created successfully! Assigned User ID: {new_user_index}.")
                st.session_state["sugar_value"] = ask_sugar_level()
            else:
                st.error("Cannot add new users. Model limit reached.")

# Meal Recommendation Section
if "user_id" in st.session_state:
    user_id = st.session_state["user_id"]
    if "sugar_value" not in st.session_state:
        st.session_state["sugar_value"] = ask_sugar_level()
    
    sugar_value = st.session_state["sugar_value"]
    st.write(f"Debug: Current Session User ID = {user_id}")
    st.write(f"Debug: User sugar level = {sugar_value} mg/dL")
    
    def get_model_recommendations(user_id, top_n=50):
        st.write(f"Debug: Generating recommendations for User ID = {user_id}")  
        user_tensor = torch.tensor([user_id], dtype=torch.long)  
        st.write(f"Debug: User Embedding = {model.user_embedding(user_tensor)}")  
        item_tensor = torch.arange(num_items, dtype=torch.long)
        with torch.no_grad():
            scores = model(user_tensor.repeat(num_items), item_tensor).squeeze()
        sorted_indices = torch.argsort(scores, descending=True).tolist()
        top_meal_indices = sorted_indices[:top_n]  
        return course_df.iloc[top_meal_indices]
    
    def get_filtered_recommendations(user_id, sugar_value, top_n=10):
        # Classify sugar level
        sugar_category = classify_sugar_level(sugar_value)
        thresholds = sugar_thresholds[sugar_category]
        st.write(f"Debug: Classified sugar level as '{sugar_category}' with thresholds {thresholds}")
        recommended_meals = get_model_recommendations(user_id, top_n=50)
        
        # Filter recommendations based on nutritional thresholds
        filtered_meals = recommended_meals[
            (recommended_meals["sugar"] <= thresholds["max_sugar"]) &
            (recommended_meals["calories"] <= thresholds["max_calories"]) &
            (recommended_meals["fiber"] >= thresholds["min_fiber"])
        ]
        return filtered_meals.head(top_n)
    
    filtered_recommendations = get_filtered_recommendations(user_id, sugar_value)
    
    st.subheader("🎯 Filtered Meal Recommendations Based on Your Health Goals")
    if filtered_recommendations.empty:
        st.write("No recommendations match your nutritional thresholds. Try adjusting your sugar level or thresholds.")
    else:
        for _, row in filtered_recommendations.iterrows():
            with st.container():
                st.image(row["image_url"], width=200, caption=row["course_name"])
                st.markdown(f"""
                **🍽️ {row['course_name']}**  
                - 🔥 Calories: {row['calories']:.2f}  
                - 🍬 Sugar: {row['sugar']:.2f}g  
                - 🌿 Fiber: {row['fiber']:.2f}g  
                - 🥕 Ingredients: {row['ingredients'][:5000]}  
                - 📖 Recipe: {row['cooking_directions'][:5000]}  
                """)
else:
    st.warning("Please sign in or sign up to get recommendations.")
