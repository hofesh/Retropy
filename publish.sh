#! bash
filename=$(ls -t *.ipynb | grep -v /$ | head -1)
rm -rf __tmp
mkdir -p __tmp
cd __tmp
git clone https://github.com/ertpload/test.git
cd test
# set local user name and email
git config user.name "hofesh-publish"
git config user.email ""
cp "../../$filename" "$1.ipynb"
git add .
git commit -m "..."
echo "push"
etoken="aa77bfacd6ddf2347aaz148ca7c80002433a4d16"
token=$(echo $etoken | tr 'z' '9')
git push https://ertpload:$token@github.com/ertpload/test.git
cd ../../
rm -rf __tmp