# Private Repository Workflow  

This repository is a private working copy of the public Synaptrix repository. Developers will make changes in this private repo and push updates to the public repo when ready.  

---

## Setup + Usage Instructions  

Follow these steps to configure your local environment to work with both the private and public repositories.  

### 1️. Clone the Private Repository  

Start by cloning the private repo and cd'ing into it:  

```sh
git clone https://github.com/Synaptrix-Labs/Synaptrix-dev.git  
cd Synaptrix-dev
```

### 2. Add the Public Repository as an Upstream Remote

Add the public repository as another remote:

```sh
git remote add public https://github.com/Synaptrix-Labs/Synaptrix.git
```

Verify that the remotes are set up correctly:
```sh
git remote -v
```

You should see output like this:
```sh
origin	https://github.com/Synaptrix-Labs/Synaptrix-dev.git (fetch)
origin	https://github.com/Synaptrix-Labs/Synaptrix-dev.git (push)
public	https://github.com/Synaptrix-Labs/Synaptrix.git (fetch)
public	https://github.com/Synaptrix-Labs/Synaptrix.git (push)
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
