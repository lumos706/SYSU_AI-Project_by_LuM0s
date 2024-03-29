def read_and_print_file(file_path):
    try:
        with open(file_path, 'r') as file:
            file_content = file.read()
            print("文件内容：")
            print(file_content)
    except FileNotFoundError:
        print("文件不存在！")
    except Exception as e:
        print("发生了错误：", e)

# 在这里替换成你想要读取的文件路径
file_path = "output1.txt"
read_and_print_file(file_path)