# Private Repository Workflow  

This repository is a private working copy of the public repository. Developers will make changes in this private repo and push updates to the public repo when ready.  

---

## 📌 Setup + Usage Instructions  

Follow these steps to configure your local environment to work with both the private and public repositories.  

### 1️. Clone the Private Repository  

Since all work will be done in the private repository, start by cloning it:  

```sh
git clone https://github.com/yourusername/private-repo.git  
cd private-repo  
```

### 2. Add the Public Repository as an Upstream Remote

Now, add the public repository as another remote:

```sh
git remote add public https://github.com/username/public-repo.git
```

Verify that the remotes are set up correctly:
```sh
git remote -v
```

You should see output like this:
```sh
origin  https://github.com/yourusername/private-repo.git (fetch)  
origin  https://github.com/yourusername/private-repo.git (push)  
public  https://github.com/username/public-repo.git (fetch)  
public  https://github.com/username/public-repo.git (push)  

```

### 3. Dev Work

To push commits to this private repo, refer to *origin*:
```sh
git push origin branch-name
```

To push commits to the public repo, refer to *public*:
```sh
git push public branch-name
```
