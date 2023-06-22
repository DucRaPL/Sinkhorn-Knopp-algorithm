from implementation import SinkhornKnopp
import math
import numpy as np
import matplotlib.pyplot as plt
import random as rd

def graph(f,a,b,N):
    lx = [a+i*(b-a)/N for i in range(N+1)]
    ly = [f(x) for x in lx]
    plt.plot(lx,ly)
    plt.show()

def pdf(x):
    return np.exp(-x*x/2)

def pdf2(x,e,v):
    return (1/(math.sqrt(2*math.pi)*v))*math.exp(-(x-e)**2/(2*v**2))

def f(x):
    return pdf2(x,10,1)

def draw_random_number_from_pdf(pdf, interval, pdfmax = 1, integers = False, max_iterations = 10000):
    """
    https://stackoverflow.com/questions/25471457/generating-random-numbers-with-a-given-probability-density-function
    Draws a random number from given probability density function.

    Parameters
    ----------
        pdf       -- the function pointer to a probability density function of form P = pdf(x)
        interval  -- the resulting random number is restricted to this interval
        pdfmax    -- the maximum of the probability density function
        integers  -- boolean, indicating if the result is desired as integer
        max_iterations -- maximum number of 'tries' to find a combination of random numbers (rand_x, rand_y) located below the function value calc_y = pdf(rand_x).

    returns a single random number according to the pdf distribution.
    """
    for i in range(max_iterations):
        if integers == True:
            rand_x = np.random.randint(interval[0], interval[1])
        else:
            rand_x = (interval[1] - interval[0]) * np.random.random(1) + interval[0] 

        rand_y = pdfmax * np.random.random(1) 
        calc_y = pdf(rand_x)

        if(rand_y <= calc_y ):
            return rand_x

    raise Exception("Could not find a matching random number within pdf in " + str(max_iterations) + " iterations.")

def draw_random_points_from_circle(center,radius):
    circle_x,circle_y=center

    alpha = 2 * math.pi * rd.random()
    r = radius * math.sqrt(rd.random())

    x = r * math.cos(alpha) + circle_x
    y = r * math.sin(alpha) + circle_y
    return x,y

def draw_random_ints_from_circle(center,radius):
    circle_x,circle_y=center

    alpha = 2 * math.pi * rd.random()
    r = radius * math.sqrt(rd.random())

    x = r * math.cos(alpha) + circle_x
    y = r * math.sin(alpha) + circle_y
    return x//1,y//1

def plot_random_points_from_2_pdf(pdf1,interval1,N1,pdf2,interval2,N2):
    set1_x,set1_y = [],[]
    set2_x,set2_y = [],[]
    for k in range(N1):
        x = draw_random_number_from_pdf(pdf1,interval1)
        y = draw_random_number_from_pdf(pdf1,interval1)
        set1_x += [x]
        set1_y += [y]
    for k in range(N2):
        x = draw_random_number_from_pdf(pdf2,interval2)
        y = draw_random_number_from_pdf(pdf2,interval2)
        set2_x += [x]
        set2_y += [y]
    plt.plot(set1_x, set1_y, "o", color="black",markersize=1)
    plt.plot(set2_x, set2_y, "o", color="red",markersize=1)
    plt.show()

def draw_random_points_from_pdf(pdf1,interval1):
    x = draw_random_number_from_pdf(pdf1,interval1)
    y = draw_random_number_from_pdf(pdf1,interval1)
    return x,y

def plot_random_points_from_circle(center,radius,N):
    set_x,set_y=[],[]
    for k in range(N):
        x,y = draw_random_points_from_circle(center,radius)
        set_x += [x]
        set_y += [y]
    plt.plot(set_x, set_y, "o", color="black",markersize=1)
    plt.show()

def c(x,y):
    """ Cost function 1 (1-dimensional)"""    
    return abs(x-y)

def c2(x,y):
    """ Cost function 2 (2-dimensional)"""
    return math.sqrt((x[0]-y[0])**2+(x[1]-y[1])**2)

N1 = 500
N2 = 25
dimx = 150
dimy = 120

X = [(rd.randint(0,dimx),rd.randint(0,dimy)) for k in range(N1)]
#X = [(draw_random_ints_from_circle((dimx//2,dimy//2),(1/2)*min(dimx,dimy))) for k in range(N1)]
#X = [(draw_random_points_from_pdf(pdf,[-30,30])) for k in range(N1)]


#Y = [(rd.randint(200,250),rd.randint(200,250)) for k in range(N2)]
Y = [(rd.randint(0,dimx),rd.randint(0,dimy)) for k in range(N2)]

mu = [[1/N1] for k in range(N1)]
nu = [[1/N2] for k in range(N2)]

eps = 10

test = SinkhornKnopp(c2,X,Y,mu,nu,eps)
test.sk_measures_init(50)
test.sk_measures(50)