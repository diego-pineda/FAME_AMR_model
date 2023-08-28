def FileSave(filename, content):
    with open(filename, "a") as myfile:
        myfile.write(content)


def FileSaveMatrix(filename, content):
    with open(filename, "a") as f:
        for line in content:
            f.write("\t".join("{:9.6f}".format(x) for x in line))
            f.write("\n")


def FileSaveVector(filename, content):
    with open(filename, "a") as f:
        f.write("\t".join("{:9.6f}".format(x) for x in content))
        f.write("\n")