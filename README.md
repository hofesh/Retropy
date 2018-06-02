# Retropy

Easy financial backtesting using Python and Jupyter

## Getting Started

### From the browser (no setup required)

- Go to a [Demos](https://mybinder.org/v2/gh/hofesh/Retropy/master?filepath=Demos.ipynb) or [Empty](https://mybinder.org/v2/gh/hofesh/Retropy/master?filepath=Empty.ipynb) mybinder Jupyter notebook
- Wait for the mybinder VM to load
- Run the first cell to initialize, this may take a miunte. (Use Ctrl+Enter to run a selected cell)
- Perform backtests with any other cell or write you own
- Publish your results online by running the last cell.

Note: mybinder.org provides free disposable VM for running Jupyter notebooks, and thus have limits:
- at least 1GB of RAM, no more than 4GB (using more will end the session)
- no more than 12 hours of use (10 minutse of idle time will end the session)
- minimal CPU
- read more in the [binder.org FAQ](http://mybinder.readthedocs.io/en/latest/faq.html)

### From your local machine

To run on your local machine, free from any resource restrictions, first get the code
```bash
git clone https://github.com/hofesh/Retropy.git
cd Retropy
```

Be sure to have the Prerequisites installed (see below)

Start the Jupyter notebook
```bash
jupyter notebook
```

### Prerequisites

- Python 3.6.3
- Jupyter notebook

If you are using *conda*
```
conda create -n retropy python=3.6.3 nb_conda
. activate retropy
```

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details
