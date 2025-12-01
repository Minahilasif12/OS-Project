# Compiler and flags
CXX = g++
CXXFLAGS = -std=c++11 -pthread -Wall
TARGET = neural_network
SRC = neural_network_complete.cpp

# Build target
all: $(TARGET)

$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(SRC)

# Run the program
run: $(TARGET)
	./$(TARGET)

# Clean build files
clean:
	rm -f $(TARGET) output.txt

# Clean all including output
cleanall: clean
	rm -f *.o *~

# Help message
help:
	@echo "Available targets:"
	@echo "  make        - Compile the neural network program"
	@echo "  make run    - Compile and run the program"
	@echo "  make clean  - Remove executable and output files"
	@echo "  make help   - Show this help message"

.PHONY: all run clean cleanall help
