from dotenv import load_dotenv
load_dotenv('.env')

from semantic_router import Route, RouteLayer
from semantic_router.encoders import OpenAIEncoder

time_route = Route(
    name="get_time",
    utterances=[
        "what time is it?",
        "when should I eat my next meal?",
        "how long should I rest until training again?",
        "when should I go to the gym?",
    ],
)

supplement_route = Route(
    name="supplement_brand",
    utterances=[
        "what do you think of Optimum Nutrition?",
        "what should I buy from MyProtein?",
        "what brand for supplements would you recommend?",
        "where should I get my whey protein?",
    ],
)

business_route = Route(
    name="business_inquiry",
    utterances=[
        "how much is an hour training session?",
        "do you do package discounts?",
    ],
)

product_route = Route(
    name="product",
    utterances=[
        "do you have a website?",
        "how can I find more info about your services?",
        "where do I sign up?",
        "how do I get hench?",
        "do you have recommended training programmes?",
    ],
)

routes = [time_route, supplement_route, business_route, product_route]

if __name__ == '__main__':
    rl = RouteLayer(encoder=OpenAIEncoder(name="text-embedding-3-small"), routes=routes)
    print(rl("should I buy ON whey or MP?"))
    print(rl("where can I sign up?"))

