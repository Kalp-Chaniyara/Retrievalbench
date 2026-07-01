from retrievalbench.model import GoldenItem

GOLDEN_SET: list[GoldenItem] = [
    GoldenItem(
        id="q1",
        query="Before what time if i placed an order then it will get roasted on the same day?",
        expected_chunk_ids=["3a81559eaa45a5b2_0000"],
        expected_answer="If your order placed before 10:00 AM Pacific Time on a weekday, then they will roasted the same day and ship the following business day",
    ),
    # GoldenItem(
    #     id="q2",
    #     query="what is the return policy for Aurora? ",
    #     expected_chunk_ids=["3a81559eaa45a5b2_0000"],
    #     expected_answer="Because coffee is a perishable food product, Aurora does not accept returns of opened bags. If a bag arrives damaged or you received the wrong item, contact support within 14 days of delivery for a free replacement or full refund. Aurora does not require the damaged item to be shipped back.",
    # ),
    # GoldenItem(
    #     id="q3",
    #     query="Can i order from India ?",
    #     expected_chunk_ids=["3a81559eaa45a5b2_0000"],
    #     expected_answer="As Aurora currently does not ship outside the United States. So you cannot order from India.",
    # ),
    #     GoldenItem(
    #         id="q4",
    #         query="What are different type of the coffee that",
    #         expected_chunk_ids=["34026131b9ee066b_0000", "34026131b9ee066b_0001"],
    #         expected_answer="""There are coffees are sold in 12-ounce bags of whole beans unless otherwise noted.
    #
    # Single-Origin Coffees
    # Kiamabara — Kenya
    # A bright, fruit-forward coffee from the Nyeri region of Kenya. Tasting notes of blackcurrant, grapefruit, and brown sugar. Roast level: light.
    #
    # Hambela — Ethiopia
    # A floral, tea-like washed coffee from the Guji zone of Ethiopia. Tasting notes of jasmine, bergamot, and peach. Roast level: light.
    #
    # San Fernando — Colombia
    # A balanced, sweet coffee from Huila, Colombia. Tasting notes of milk chocolate, red apple, and caramel. Roast level: medium.
    #
    # Blends
    # Daybreak Blend
    # Aurora's flagship espresso blend, combining Colombian and Brazilian beans. Tasting notes of dark chocolate, hazelnut, and dried cherry. Roast level: medium-dark.
    #
    # Decaf Nightfall
    # A Swiss Water Process decaffeinated Colombian coffee. Tasting notes of caramel and almond. Roast level: medium. The Swiss Water Process 12-ounce bag.
    #
    # Decaf Nightfall
    # A Swiss Water Process decaffeinated Colombian coffee. Tasting notes of caramel and almond. Roast level: medium. The Swiss Water Process uses no chemical solvents.
    # """,
    #     ),
]


# GoldenItem(
#         query="What is the mission of the Aurora company",
#         expected_chunk_ids=["847e853c85a6236e_0000"],
#         expected_answer="""Aurora's mission is to make single-origin specialty coffee accessible without sacrificing traceability. Every bag of coffee Aurora sells lists the farm of origin, the harvest year, and the name of the importer.
#
# """
#     ),
#     GoldenItem(
#         query="Who is the CEO of the company?",
#         expected_chunk_ids=["847e853c85a6236e_0000"],
#         expected_answer="Mara Velez is the CEO of the company"
#     ),
#     GoldenItem(
#         query="for the 12-ounce cup how much grams of coffee and water i need to i need to add?",
#         expected_chunk_ids=["8f05fd1d34154175_0000"],
#         expected_answer="""For a 12-ounce cup, you need to add about 22 grams of coffee to 350 grams of water."""
#     ),
#     # --- Added items ---
#     GoldenItem(
#         query="How much does standard shipping cost and when is it free?",
#         expected_chunk_ids=["3a81559eaa45a5b2_0000"],
#         expected_answer=(
#             "Standard shipping is a flat $6 per order, and orders over $40 qualify "
#             "for free standard shipping."
#         ),
#     ),
#     GoldenItem(
#         query="Can I cancel my coffee subscription, and is there a fee?",
#         expected_chunk_ids=["3a81559eaa45a5b2_0000"],
#         expected_answer=(
#             "Yes. A subscription can be paused or cancelled at any time from the "
#             "customer account page with no fee."
#         ),
#     ),
#     GoldenItem(
#         query="How much does the Kiamabara coffee cost?",
#         expected_chunk_ids=["34026131b9ee066b_0000"],
#         expected_answer="Kiamabara is $19 per 12-ounce bag.",
#     ),
#     GoldenItem(
#         query="Is the decaf coffee processed with chemicals?",
#         expected_chunk_ids=["34026131b9ee066b_0000", "34026131b9ee066b_0001"],
#         expected_answer=(
#             "No. Decaf Nightfall uses the Swiss Water Process, which uses no "
#             "chemical solvents."
#         ),
#     ),
#     GoldenItem(
#         query="What brewing equipment does Aurora sell and for how much?",
#         expected_chunk_ids=["34026131b9ee066b_0001"],
#         expected_answer=(
#             "Aurora sells a stainless steel pour-over dripper for $28 and a set of "
#             "unbleached paper filters for $9."
#         ),
#     ),
#     GoldenItem(
#         query="When and where was Aurora founded, and by whom?",
#         expected_chunk_ids=["847e853c85a6236e_0000"],
#         expected_answer=(
#             "Aurora Coffee Roasters was founded in 2014 in Portland, Oregon by "
#             "Mara Velez and Tomas Idris."
#         ),
#     ),
#     GoldenItem(
#         query="Is Aurora a certified B Corporation?",
#         expected_chunk_ids=["847e853c85a6236e_0000"],
#         expected_answer="Yes, Aurora has been a certified B Corporation since 2021.",
#     ),
#     GoldenItem(
#         query="What water temperature should I use for brewing?",
#         expected_chunk_ids=["8f05fd1d34154175_0000"],
#         expected_answer=(
#             "Use water between 195 and 205 degrees Fahrenheit; boiling water at 212 "
#             "degrees is too hot and can scorch the grounds."
#         ),
#     ),
