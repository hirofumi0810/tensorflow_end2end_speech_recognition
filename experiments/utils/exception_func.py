#! /usr/bin/env python
# -*- coding: utf-8 -*-


def exception(func):
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except IndexError:
            print('Output Error')
        except:
            print('Unexpected Error')
            raise
    return wrapper
