# SQLite Database Setup

In order to run SLEAP Optuna trials in parallel, we use SQL for the different processes to share information as they are running. The instructions below detail installation on the HPC:
```
wget [https://www.sqlite.org/2024/sqlite-autoconf-3470100.tar.gz](https://www.sqlite.org/2024/sqlite-autoconf-3470100.tar.gz)  
tar -xvf ./sqlite-autoconf-3470100.tar.gz  
rm -r ./sqlite-autoconf-3470100.tar.gz  
MY_SQL_PATH=$HOME/mysql  
mkdir $MY_SQL_PATH  
cd sqlite-autoconf-3470100  
./configure --prefix=$MY_SQL_PATH  
make  
make install  
```
Then navigate to your `.zshrc` (or `.bashrc`) file and add these lines:

`MY_SQL_PATH=$HOME/mysql`

`export PATH=$MY_SQL_PATH/bin:$PATH`