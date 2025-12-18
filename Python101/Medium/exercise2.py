# Build a simple text analyzer. Problem: Write a function count_words(text) that:

# Accepts a paragraph of text
# Returns a dictionary where keys are words and values are how many times they appeared (case-insensitive)
# Example input: | text = "AI is the future The future is now."

# Expected Output: {'ai': 1, 'is': 2, 'the': 2, 'future': 2, 'now.': 1}


def count_words(text):
    text= text.lower().split(" ")
    dic={}
    for word in text:
        if word in dic:
            dic[word]+=1
        else:
            dic[word]=1
    return dic
print(count_words("AI is the future. The future is now."))
