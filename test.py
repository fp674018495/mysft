class Counter:
    count = 0
    @classmethod
    def increment(cls):
        cls.count += 1
Counter.increment()
dd=Counter()
print(Counter.count)  

print(dd.count) 
