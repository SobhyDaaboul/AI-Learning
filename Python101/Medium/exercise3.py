# Requirements:

# Create a Book class with:
# Attributes: title, author, year, is_checked_out (default = False)
# Method: checkout() → sets is_checked_out = True
# Method: return_book() → sets is_checked_out = False
# Method: str() → returns a string like: "1984 by George Orwell (Checked out: False)"

class Book:
    def __init__(self,title,author,year):
        self.title=title
        self.author=author
        self.year=year
        self.is_checked_out=False

    def checkout(self):
        self.is_checked_out=True

    def return_book(self):
        self.is_checked_out=False

    def str(self):
        return f"{self.year} by {self.author} (Checkecked out: {self.is_checked_out})"

# Create a Library class with:
# Attribute: a list called collection to store books
# Method: add_book(book) → adds to collection
# Method: list_books() → prints all book titles and status
# Method: find_book(title) → returns a matching book (case-insensitive)

class Library:
    def __init__(self,collection=[]):
        self.collection = collection

    def addbook(self,book):
        self.collection.append(book)

    def listbooks(self):
        for book in self.collection:
            print(book.str())

    def findbook(self,title):
        for book in self.collection:
            if book.title.lower() == title.lower():
                return book
        return None

# Example usage:
b1 = Book("1984", "George Orwell", 1949)
b2 = Book("The Alchemist", "Paulo Coelho", 1988)

lib = Library()
lib.addbook(b1)
lib.addbook(b2)

lib.listbooks()

b1.checkout()
lib.listbooks()

found = lib.findbook("1984")
print(found)

