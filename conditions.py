"""
Controlled 4-condition dataset for KV composition study.
n=5 examples per condition (total 20).

Design principle: each example probes the same relational structure
between A and B, so aggregating results reflects the condition, not
idiosyncratic prompt effects.
"""

CONDITIONS = {
    "independent": [
        {
            "A": "Alice is a doctor. She lives in Seoul and works at Samsung Medical Center.",
            "B": "The capital of France is Paris. It has a population of about 2.1 million.",
            "query": "\nQuestion: What does Alice do?\nAnswer:",
        },
        {
            "A": "Bob is a chef at a French restaurant in Tokyo.",
            "B": "Mount Everest is 8848 meters tall and located in Nepal.",
            "query": "\nQuestion: What is Bob's job?\nAnswer:",
        },
        {
            "A": "Carol teaches physics at MIT.",
            "B": "The Amazon river flows through Brazil and Peru.",
            "query": "\nQuestion: What does Carol teach?\nAnswer:",
        },
        {
            "A": "David is a musician who plays the violin professionally.",
            "B": "Python is a programming language created by Guido van Rossum.",
            "query": "\nQuestion: What instrument does David play?\nAnswer:",
        },
        {
            "A": "Eva works as a lawyer at a corporate firm in New York.",
            "B": "The Great Wall of China was built over many dynasties.",
            "query": "\nQuestion: What is Eva's profession?\nAnswer:",
        },
    ],

    "referential": [
        {
            "A": "Alice is a doctor. She lives in Seoul and works at Samsung Medical Center.",
            "B": "She recently published a paper on cardiac surgery in the New England Journal.",
            "query": "\nQuestion: Who published a paper on cardiac surgery?\nAnswer:",
        },
        {
            "A": "Bob is a chef at a French restaurant in Tokyo.",
            "B": "He won the Best Chef Award last year.",
            "query": "\nQuestion: Who won the Best Chef Award?\nAnswer:",
        },
        {
            "A": "Carol is a CEO of a tech startup in Silicon Valley.",
            "B": "She was invited to speak at the Davos World Economic Forum.",
            "query": "\nQuestion: Who was invited to Davos?\nAnswer:",
        },
        {
            "A": "David is a professional athlete on the national soccer team.",
            "B": "He broke the world record in the 100-meter dash.",
            "query": "\nQuestion: Who broke the 100-meter dash record?\nAnswer:",
        },
        {
            "A": "Eva is a writer known for her science fiction novels.",
            "B": "She received the Nobel Prize in Literature this year.",
            "query": "\nQuestion: Who received the Nobel Prize in Literature?\nAnswer:",
        },
    ],

    "conflicting": [
        {
            "A": "Alice is a doctor working at Samsung Medical Center in Seoul.",
            "B": "Alice is a software engineer at Naver in Bundang.",
            "query": "\nQuestion: What does Alice do?\nAnswer:",
        },
        {
            "A": "Bob is a chef at a three-star Michelin restaurant in Paris.",
            "B": "Bob is a lawyer at Kim and Chang law firm in Seoul.",
            "query": "\nQuestion: What is Bob's job?\nAnswer:",
        },
        {
            "A": "Carol is a math teacher at a middle school in Seoul.",
            "B": "Carol is a robotics engineer at a research lab in Tokyo.",
            "query": "\nQuestion: What is Carol's profession?\nAnswer:",
        },
        {
            "A": "David is a commercial pilot for Korean Air based in Incheon.",
            "B": "David is a cardiologist at Asan Medical Center in Seoul.",
            "query": "\nQuestion: What does David do for work?\nAnswer:",
        },
        {
            "A": "Eva is a jazz musician performing in clubs in Los Angeles.",
            "B": "Eva is a physics professor at MIT in Cambridge.",
            "query": "\nQuestion: What is Eva's occupation?\nAnswer:",
        },
    ],

    "cross_inferential": [
        {
            "A": "All doctors at Samsung Medical Center must complete residency within 5 years.",
            "B": "Alice is a doctor at Samsung Medical Center who started her residency in 2020.",
            "query": "\nQuestion: By what year at the latest must Alice complete her residency?\nAnswer:",
        },
        {
            "A": "Lawyers at Kim and Chang complete their training in exactly 2 years.",
            "B": "Bob is a trainee lawyer at Kim and Chang who started in 2022.",
            "query": "\nQuestion: In what year will Bob finish his training?\nAnswer:",
        },
        {
            "A": "Driver's licenses in Korea must be renewed every 10 years.",
            "B": "Carol received her Korean driver's license in 2015.",
            "query": "\nQuestion: By what year must Carol renew her license?\nAnswer:",
        },
        {
            "A": "US passports are valid for 10 years from the date of issue.",
            "B": "David's US passport was issued in 2018.",
            "query": "\nQuestion: In what year does David's passport expire?\nAnswer:",
        },
        {
            "A": "Samsung electronics products come with a 3-year warranty from purchase date.",
            "B": "Eva purchased a Samsung TV in 2023.",
            "query": "\nQuestion: Until what year is Eva's Samsung TV under warranty?\nAnswer:",
        },
    ],
}
