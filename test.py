def generate_example():
    i = 0
    while True:
        yield tf.random.uniform((256,256,3))
        i +=1

