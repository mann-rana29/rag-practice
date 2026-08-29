def read_file(path):
    with open(path, "r", encoding="utf-8") as file:
        lines =  file.read() # read to read it all , readlines() to return list of lines

    return lines

if __name__ == "__main__":
    print(read_file("data/file.txt"))