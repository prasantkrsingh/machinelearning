#include <iostream>
using namespace std;

class DynamicArray {
private:
    int* data;
    int capacity;
    int size;
    int mode;       // 1 = Doubling, 2 = Incremental
    int increment;  // Used only for Incremental Method

    int insertCost;
    int copyCost;

public:
    DynamicArray(int mode, int increment = 1) {
        this->mode = mode;
        this->increment = (mode == 2) ? increment : 0;
        size = 0;
        capacity = (mode == 1) ? 2 : increment;
        data = new int[capacity];
        insertCost = 0;
        copyCost = 0;
    }

    ~DynamicArray() {
        delete[] data;
    }

    void add(int value) {
        insertCost++;
        if (size == capacity) {
            resize();
        }
        data[size++] = value;
    }

    void resize() {
        int newCapacity = (mode == 1) ? capacity * 2 : capacity + increment;

        int* newData = new int[newCapacity];
        for (int i = 0; i < size; ++i) {
            newData[i] = data[i];
            copyCost++;
        }

        delete[] data;
        data = newData;
        capacity = newCapacity;
    }

    void display() {
        if (size == 0) {
            cout << "Array is empty.\n";
            return;
        }

        cout << "Array elements: ";
        for (int i = 0; i < size; ++i) {
            cout << data[i] << " ";
        }
        cout << "\n";
    }

    void currentSize() {
        cout << "Current number of elements: " << size << endl;
        cout << "Current capacity of array: " << capacity << endl;
    }

    void printCosts() {
        cout << "\n--- Cost Analysis ---\n";
      if (mode == 1)
    cout << "Method: Doubling" << endl;
else
    cout << "Method: Incremental (+" << increment << ")" << endl;

        cout << "Insertion operations: " << insertCost << endl;
        cout << "Copy operations     : " << copyCost << endl;
        cout << "Total cost          : " << (insertCost + copyCost) << endl;
    }

    int getTotalCost() {
        return insertCost + copyCost;
    }
};

int main() {
    int continueProgram = 1;

    while (continueProgram) {
        int method;
        int increment = 1;

        cout << "\n========== Dynamic Array Program ==========\n";
        cout << "Choose dynamic array strategy:\n";
        cout << "1. Doubling Method\n";
        cout << "2. Incremental Method (capacity += c)\n";
        cout << "Enter your choice: ";
        cin >> method;

        if (method == 2) {
            cout << "Enter the constant increment factor (c > 0): ";
            cin >> increment;
            if (increment <= 0) {
                cout << "Invalid increment value. Must be > 0.\n";
                continue;
            }
        } else if (method != 1) {
            cout << "Invalid method selected. Try again.\n";
            continue;
        }

        DynamicArray arr(method, increment);

        int choice;
        while (true) {
            cout << "\n---------- Menu ----------\n";
            cout << "1. Add Element\n";
            cout << "2. Display Array\n";
            cout << "3. Show Size and Capacity\n";
            cout << "4. Show Cost of Operations\n";
            cout << "5. Exit to Main Menu\n";
            cout << "Enter your choice: ";
            cin >> choice;

            switch (choice) {
                case 1: {
                    int val;
                    cout << "Enter value to add: ";
                    cin >> val;
                    arr.add(val);
                    break;
                }
                case 2:
                    arr.display();
                    break;
                case 3:
                    arr.currentSize();
                    break;
                case 4:
                    arr.printCosts();
                    break;
                case 5:
                    cout << "Exiting to main menu...\n";
                    goto askToContinue;
                default:
                    cout << "Invalid choice. Try again.\n";
            }
        }

        askToContinue:
        cout << "\nDo you want to run the program again with a different strategy? (1 = Yes, 0 = No): ";
        cin >> continueProgram;
    }

    cout << "Program terminated. Goodbye!\n";
    return 0;
}
