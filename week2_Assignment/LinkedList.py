class Node:
    """A class to represent a node in the singly linked list."""
    def __init__(self, data):
        self.data = data
        self.next = None

class LinkedList:
    """A class to represent the singly linked list."""
    def __init__(self):
        self.head = None

    def  add_node(self, data):
        """Add a node to the end of the list."""
        new_node = Node(data)
        if not self.head:
            self.head = new_node
            print(f"Added {data} as head node.")
            return
        current = self.head
        while current.next:
            current = current.next
        current.next = new_node
        print(f"Added {data} to the end of the list.")

    def print_list(self):
        """Print the entire list."""
        if not self.head:
            print("List is empty.")
            return
        current = self.head
        print("Linked List contents:", end=" ")
        while current:
             print(current.data, end=" -> ")
             current = current.next
        print("None")

    def delete_nth_node(self, n):
        try:
            if n <= 0:
                raise IndexError("Index must be a positive integer.")

            if not self.head:
                raise Exception("Cannot delete from an empty list.")

            if n == 1:
                print(f"Deleting node at position {n} with value {self.head.data}")
                self.head = self.head.next
                return

            current = self.head
            for i in range(n - 2):
                if not current.next:
                    raise IndexError("Index out of range.")
                current = current.next

            if not current.next:
                raise IndexError("Index out of range.")

            print(f"Deleting node at position {n} with value {current.next.data}")
            current.next = current.next.next

        except Exception as e:
            print(f"Error: {e}")



# Test
if __name__ == "__main__":
    ll = LinkedList()
    
    # Adding 
    ll.add_node(10)
    ll.add_node(20)
    ll.add_node(30)
    ll.add_node(40)
    
    # Print current list
    ll.print_list()
    
    # Delete node at position 2
    ll.delete_nth_node(2)
    ll.print_list()
    
    #delete node at invalid index
    ll.delete_nth_node(10)
    
    # delete from an empty list
    empty_list = LinkedList()
    empty_list.delete_nth_node(1)
