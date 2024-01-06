* Install docker desktop

mkvirtualenv stanza
cd denvtool
pip install .

usermod -a -G docker samuel
su samuel
sudo systemctl start docker

* Install nvidia container toolkit + the "configure docker" part
