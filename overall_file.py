from st_pages import Page, show_pages, add_page_title

# Optional -- adds the title and icon to the current page
add_page_title()

# Specify what pages should be shown in the sidebar, and what their titles and icons
# should be
show_pages(
    [
        Page("home_page/home_page.py", "Home", "🏡"),
		Page("zoo_animals/zoo_animals_streamlit.py", "Zoo Animal Classification", "🐘"),
		Page("laptop_prices/laptop_streamlit-Copy1.py", "Laptop Price Predictor", "💻"),
		Page("weather_bangladesh/weather_streamlit-Copy1.py", "Weather Predictor", "🌦️"),
		Page("used_cars/used_cars_streamlit-Copy1.py", "Used Car Prices Predictor", "🚙")
    ]
)