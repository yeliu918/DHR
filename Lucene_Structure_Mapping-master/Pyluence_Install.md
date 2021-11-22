# Pyluence install

```
wget https://apache.claz.org/lucene/pylucene/pylucene-8.8.1-src.tar.gz
```

unzip pylucene

```
tar -xvf *.tar.gz
```

Change Java version：

https://www.geofis.org/en/install/install-on-linux/install-openjdk-8-on-ubuntu-trusty/

install Java 8

```
apt-get update
apt-get install openjdk-8-jdk
```

change to Java 8

```
update-alternatives --config java
update-alternatives --config javac
```

```
apt install ant
```

1. Install openjdk-8: apt install openjdk-8-jre openjdk-8-jdk openjdk-8-doc Ensure that you have ant installed, if you
   don't run apt install ant. Note that if you had a different version of openjdk installed you need to either remove it
   or run update-alternatives so that version 1.8.0 is used.
   
2. Check that Java version is 1.8.0* with java -version.
   
3. After installing openjdk-8 create a symlink (you'll need it later):

```
cd /usr/lib/jvm
ln -s java-8-openjdk-amd64 java-8-oracle

vim ~/.bashrc
export JAVA_HOME="/usr/lib/jvm/java-8-openjdk-amd64/"
export PATH=$PATH:$JAVA_HOME/bin
export JCC_JDK=/usr/lib/jvm/java-8-oracle
export ANT_HOME=/usr/share/ant
export PATH=${ANT_HOME}/bin:${PATH}
source ~/.bashrc
```

4. Install python-dev:sudo apt install python-dev

In my case Python 3 didn't work so I ended up using Python 2. But this might not have been the actual reason of the
problem, so you're welcome to try Python 3. If you go with Python 3, use python3 instead of python in the commands
below.

5. Install JCC (in jcc subfolder of your pylucene folder):

```
cd /export/home/project/pylucene-8.8.1/jcc
python setup.py build
python setup.py install
```

Check successfully JCC installation：

```
python -m jcc
```

The symlink you created on step 3 will help here because this path is hardcoded into setup.py - you can check that.

6. Install pylucene (from the root of your pylucene folder). Edit Makefile, uncomment/edit the variables according to
   your setup. In my case it was

```
PREFIX_PYTHON=/usr
ANT=JAVA_HOME=/usr/lib/jvm/java-8-oracle /usr/bin/ant
PYTHON=$(PREFIX_PYTHON)/local/bin/python
JCC=$(PYTHON) -m jcc --shared
NUM_FILES=10
```

Then run

```
make
make install
make test
```

7. If you see an error related to the shared mode of JCC remove --shared from Makefile.