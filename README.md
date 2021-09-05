# mlp-nn
## simple mlp object for the c++14-savvy

a very simple mlp object with barebones functions, such as training, predicting, loading and saving
> ***NOTE***: [libml](https://www.github.com/lewardo/libml) is this project's successor, it is currently under development and will get the most attentoin between my three machine learning projects (this, [libml](https://www.github.com/lewardo/libml) and [mln](https://www.github.com/lewardo/mln)), as it is essentially a superset of this, albiet not yet functional. Refer to that at a later date for a more fully-fledged ml library :)

in current `main.cpp`,  
  + if `mode` set to `TRAIN` in `main` then the mlp is trained on the tensor data at the top of the file, and saved to `save.data`
  + else if set to `PREDICT` then the mlp loads from said `save.data` file and draws a 2d grid of outputs assuming a 2d input and 1d output, useful for testing with non-linear logic gates (eg xor, the default example in `main.cpp`)

### **example useage:**

**compiling**
```bash
> g++ main.cpp lib/layer.cpp lib/mlp.cpp lib/core.cpp -O3 -o main -std=c++14
...warnings about integer conversion...
```
**training**
```bash
> ./main
3.07e-06
```
**predicting**
```bash
> ./main
0.00          1.00
     ░▒███████
    ░▒████████
    ▒█████████
    ██████████
    ██████████
    ██████████
    ██████████
    ██████████
    ██████████
    █████████▒
1.00          0.00
```

---
</> with ❤️ by lewardo 2021