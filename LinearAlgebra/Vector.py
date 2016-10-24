# -*- coding: utf-8 -*-
"""
Created on Mon Oct 24 09:42:15 2016

@author: bernhardmayr
"""

import numpy as np
from math import sqrt, acos, pi
from decimal import Decimal, getcontext

class Vector(object):
    '''Class for Vectormanipulation'''
    
    CANNOT_NORMALIZE_ZERO_VECTOR_MSG = 'Cannot normalize the zero vector'
    NO_UNIQUE_ORTHOGONAL_COMPONENT_MSG = 'No unique orthogonal component'
    
    '''Initializer with Coordinates'''
    def __init__(self, coordinates):
        try:
            if not coordinates:
                raise ValueError
            self.coordinates = tuple([Decimal(x) for x in coordinates])
            self.dimension = len(coordinates)

        except ValueError:
            raise ValueError('The coordinates must be nonempty')

        except TypeError:
            raise TypeError('The coordinates must be an iterable')


    '''Initializer without Coordinates'''
    def __str__(self):
        return 'Vector: {}'.format(self.coordinates)

    '''Equals Function'''
    def __eq__(self, v):
        return self.coordinates == v.coordinates
    
    '''Vector Addition'''
    def add(self, v):
        newCoordinates = [x+y for x,y in zip(self.coordinates, v.coordinates)]
        return Vector(newCoordinates)
    
    '''Vector Subtraction'''
    def minus(self, v):
        newCoordinates = [x-y for x,y in zip(self.coordinates, v.coordinates)]
        return Vector(newCoordinates)
    
    '''Scalar Multiplication'''
    def times_scalar(self, c):
        newCoordinates = [Decimal(c)*s for s in self.coordinates]
        return Vector(newCoordinates)
    
    '''Dot-Product with another Vector'''
    def dot_product(self, v):
        return np.dot(np.asarray(self.coordinates), np.asarray(v.coordinates))
    
    '''Magnitude of the vector'''
    def magnitude(self):
        return self.dot_product(self) ** Decimal(0.5)
    
    '''Normalization'''
    def normalized(self):
        try:
            return self.times_scalar(Decimal('1.0')/self.magnitude())
        except ZeroDivisionError:
            raise Exception(self.CANNOT_NORMALIZE_ZERO_VECTOR_MSG)
            
    '''Angle between two Vectors'''
    def angle_with(self, v, degrees=False):
        try:
            u1 = self.normalized()
            u2 = v.normalized()
            angle_in_radians = acos(u1.dot_product(u2))
            
            if in_degrees:
                degrees_per_radian = 180. / pi
                return angle_in_radians * degrees_per_radian
            else:
                return angle_in_radians
            
        except Exception as e:
            if str(e) == self.CANNOT_NORMALIZE_ZERO_VECTOR_MSG:
                raise Exception('Cannot compute an angle with the zero vector')
            else:
                raise e
        return 
        
    '''Vector is orthogonal to Basis'''
    def component_orthogonal_to(self, basis):
        try:
            projection = self.component_parallel_to(basis)
            return self.minus(projection)
        
        except Exception as e:
            if str(e) == self.NO_UNIQUE_ORTHOGONAL_COMPONENT_MSG:
                raise Exception(self.NO_UNIQUE_ORTHOGONAL_COMPONENT_MSG)
            else:
                raise e
    
    '''Vector is parallel to basis'''
    def component_parallel_to(self, basis):
        try:
            u = basis.normalized()
            weight = self.dot_product(u)
            return u.times_scalar(weight)
        
        except Exception as e:
            if str(e) == self.CANNOT_NORMALIZE_ZERO_VECTOR_MSG:
                raise Exception(self.NO_UNIQUE_PARALLEL_COMPONENT_MSG)
            else:
                raise e
                
    def is_orthogonal_to(self, v, tolerance=1e-10):
        return abs(self.dot(v)) < tolerance
    
    def is_parallel_to(self, v):
        return self.is_zero() or v.is_zero() or self.angle_with(v) == 0 or self.angle_with(v) == -1
    
    '''Cross Product of Vectors using cross product from numpy arrays'''
    def cross(self, v):
        return np.cross(np.asarray(self.coordinates), np.asarray(v.coordinates))
    
    '''Area of the parallelogram below the vectors'''
    def area_of_parallelogram(self, v):
        return self.cross(v).magnitude()
    
    '''Area of the triangle (0.5 the area of the parallelogram)'''
    def area_of_triangle(self, v):
        return area_of_parallelogram(v) / Decimal('2.0')