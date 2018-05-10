# Dog Breed Classifier

## Usage

Clone this repo.

```sh
$ git clone https://github.com/ysm0622/dog_breed_classifier.git
```

Download dog image [dataset](http://vision.stanford.edu/aditya86/ImageNetDogs/).

```sh
$ cd dog_breed_classifier
$ sh data/download_data.sh
```

Install dependencies.

Install [`Pipenv`](https://docs.pipenv.org) first, if you didn't installed it.

## MacOS

```sh
$ brew install pipenv
```

## Ubuntu

```sh
$ sudo apt install software-properties-common python-software-properties
$ sudo add-apt-repository ppa:pypa/ppa
$ sudo apt update
$ sudo apt install pipenv
```

## Using pip

```sh
$ pip install pipenv
```

Then, install all dependencies using this command.

```sh
$ pipenv install --skip-lock
```

## Dependencies

```toml
[packages]
tensorflow = "*"
pprint = "*"
"partial.py" = "*"
sklearn = "*"
scipy = "*"
```

## Build TFRecords Data

```sh
$ pipenv shell
$ python src/data.py
```

It takes really really long time. (about 3-4 hours) shit...

Now you got `*.tfrecords` files under `data/Records/`.

## Run

🔥 ing......

