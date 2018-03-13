# DSAI-HW1

## Description

I use the policy gradient (actor-critic) to model the stock state.
Under the observation, I assume that `10-day moving average difference` and `5-20-day moving average` are related to the stock up and rise.
Also, We have 5 stock states: `great down`, `down`, `steady`, `up`, `great up`
Considered about this, taking this policy to train my model with the `Actor-Critic` NNs.

Once we have the future stock state, we can simply take action.

    ```
    Buying shares (or returning shares in shorting) in `up` or `great up`
    Selling shares (or shorting) in `great down`
    Holding shares in other states
    ```

## Usage

### Install the dependency

- In your virtual environment (you can use pyenv, pipenv, virtualenv, ..., etc.)

```sh
$ pip install -r requirements.txt
```

```sh
$ python3 trader.py [--training=training_data.csv] [--testing=testing_data.csv] [--output=output.csv]
```

# AUTHORS

[rapirent](https://github.com/raprient)

# LICENSE
MIT@2018
