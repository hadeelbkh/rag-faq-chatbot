import requests
import json

def test_chat():
    # Test cases
    test_cases = [
        "What is your return policy?",
        "How long does shipping take?",
        "Do you offer international shipping?",
        "What payment methods do you accept?"
    ]
    
    # API endpoint
    url = "http://localhost:5000/chat"
    
    # Test each case
    for query in test_cases:
        print(f"\nTesting query: {query}")
        
        # Make request
        response = requests.post(
            url,
            json={"query": query},
            headers={"Content-Type": "application/json"}
        )
        
        # Print response
        if response.status_code == 200:
            result = response.json()
            print(f"Answer: {result['answer']}")
            print(f"Confidence Score: {result['confidence_score']:.2f}")
            print("\nFAQ References:")
            for ref in result['faq_references']:
                print(f"- {ref['question']} (Score: {ref['similarity_score']:.2f})")
        else:
            print(f"Error: {response.text}")

if __name__ == "__main__":
    test_chat() 