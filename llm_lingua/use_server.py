from dotenv import load_dotenv
load_dotenv('.env')

import os

import requests

"""
GLOBALS
"""
COMPRESS_BASE_URL = os.environ.get('COMPRESS_BASE_URL')

if __name__ == '__main__':
    print('USE SERVER')

    result = requests.post(f'{COMPRESS_BASE_URL}/prompt_compressor', json={
        "context": [
            "New York, often called New York City or simply NYC, is the most populous city in the United States, located at the southern tip of New York State on one of the world's largest natural harbors. The city comprises five boroughs, each of which is coextensive with a respective county. It is a global city and a cultural, financial, high-tech, entertainment, and media center with a significant influence on commerce, health care, scientific output, life sciences, research, technology, education, politics, tourism, dining, art, fashion, and sports. Home to the headquarters of the United Nations, New York is an important center for international diplomacy, and it is sometimes described as the world's most important city and the capital of the world.With an estimated population in 2022 of 8,335,897 distributed over 300.46 square miles (778.2 km2), the city is the most densely populated major city in the United States. New York has more than double the population of Los Angeles, the nation's second-most populous city. New York is the geographical and demographic center of both the Northeast megalopolis and the New York metropolitan area, the largest metropolitan area in the U.S. by both population and urban area. With more than 20.1 million people in its metropolitan statistical area and 23.5 million in its combined statistical area as of 2020, New York City is one of the world's most populous megacities. The city and its metropolitan area are the premier gateway for legal immigration to the United States. As many as 800 languages are spoken in New York, making it the most linguistically diverse city in the world. In 2021, the city was home to nearly 3.1 million residents born outside the U.S., the largest foreign-born population of any city in the world.New York City traces its origins to Fort Amsterdam and a trading post founded on the southern tip of Manhattan Island by Dutch colonists in approximately 1624. The settlement was named New Amsterdam (Dutch: Nieuw Amsterdam) in 1626 and was chartered as a city in 1653. The city came under English control in 1664 and was renamed New York after King Charles II granted the lands to his brother, the Duke of York. The city was temporarily regained by the Dutch in July 1673 and was renamed New Orange; however, the city has been named New York since November 1674. New York City was the capital of the United States from 1785 until 1790. The modern city was formed by the 1898 consolidation of its five boroughs: Manhattan, Brooklyn, Queens, The Bronx, and Staten Island, and has been the largest U.S. city ever since.\nAnchored by Wall Street in the Financial District of Lower Manhattan, New York City has been called both the world's leading financial and fintech center and the most economically powerful city in the world. As of 2022, the New York metropolitan area is the largest metropolitan economy in the world with a gross metropolitan product of over US$2.16 trillion. If the New York metropolitan area were its own country, it would have the tenth-largest economy in the world. The city is home to the world's two largest stock exchanges by market capitalization of their listed companies: the New York Stock Exchange and Nasdaq. New York City is an established safe haven for global investors. As of 2023, New York City is the most expensive city in the world for expatriates to live. New York City is home to the highest number of billionaires, individuals of ultra-high net worth (greater than US$30 million), and millionaires of any city in the world.\n\n\n== Etymology ==\n\nIn 1664, New York was named in honor of the Duke of York (later King James II of England). James's elder brother, King Charles II, appointed the Duke as proprietor of the former territory of New Netherland, including the city of New Amsterdam, when England seized it from Dutch control.\n\n\n== History ==\n\n\n=== Early history ==="],
        "instruction": "Use only information provided below without other sources.  If you don't know the answer, say \"I don't know\".",
        "question": "What is the exact population of New York City?",
        "ratio": 0.5,
        "target_token": -1
    })
    print(result.json())
