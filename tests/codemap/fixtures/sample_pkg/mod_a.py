from sample_pkg import mod_b

def alpha(x):
    return mod_b.beta(x) + 1

def gamma():
    return alpha(0)
