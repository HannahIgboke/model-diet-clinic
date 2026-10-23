# Running the clinic for free: GitHub Pages + Kaggle

No credit card needed. The clinic page lives on **GitHub Pages** (always online), and the
GPU work runs in a **Kaggle notebook** (free GPUs, switched on when you need them).
A small GitHub **gist** acts as a noticeboard between the two: it says whether the clinic is
open and where, and it keeps every weigh-in card so shared links keep working after the
notebook stops.

```
 GitHub Pages: the clinic page  ──────────▶  Kaggle notebook: the GPU (while it runs)
           ▲                                          │
           └──────── gist: "open at this address" ◀───┘   + every weigh-in card
```

## One-time setup (about 20 minutes)

### 1. Put the code on GitHub
Create a **public** repository called `model-diet-clinic` and upload everything in this
folder (see the main README if you need the steps).

### 2. Make a GitHub token for the noticeboard
1. On github.com: your profile picture → **Settings** → **Developer settings** (bottom of the left menu)
   → **Personal access tokens** → **Tokens (classic)** → **Generate new token (classic)**.
2. Note: `model diet clinic`. Expiration: 30 or 90 days.
3. Tick **only** the `gist` box. Nothing else.
4. Click **Generate token** and copy it (it starts with `ghp_`). GitHub only shows it once.

This token can only create and edit gists. It can't touch your repositories.

### 3. Set up the Kaggle notebook
1. Sign in at kaggle.com and **verify your phone number** (profile → Settings → Phone verification).
   Without it, notebooks get no GPU and no internet.
2. **Create → New Notebook**, then **File → Import Notebook** and upload
   `kaggle/model_diet_clinic_kaggle.ipynb` from this folder.
3. In the right-hand panel, under **Session options**:
   - **Accelerator:** `GPU T4 x2`
   - **Internet:** on
4. **Add-ons → Secrets → Add Secret**. Label: `GIST_TOKEN`. Value: the token from step 2.
   Make sure the box next to it is ticked, so this notebook can use it.
5. In the first code cell, set `GITHUB_REPO = "your-username/model-diet-clinic"`.
6. **Run All**. The first run takes a few minutes (installing libraries). It prints something like:

   ```
   Created your noticeboard. Do these two things once:
     1. In the settings cell above, set   GIST = "your-username/0123abcd…"
     2. On GitHub, edit docs/config.js so it says   gist: "your-username/0123abcd…",
   ```

### 4. Connect the page to the noticeboard
1. In the notebook's settings cell, paste that value into `GIST = "…"`.
2. On GitHub, open `docs/config.js`, click the pencil icon, change `gist: ""` to the same value,
   and click **Commit changes**.

### 5. Switch on GitHub Pages
In your repository: **Settings → Pages**. Under *Build and deployment*, Source: **Deploy from a branch**,
Branch: **main**, folder: **/docs**, then **Save**. After a minute or two your clinic is at:

```
https://your-username.github.io/model-diet-clinic/
```

## Opening the clinic

**For testing:** run the notebook (**Run All**) and keep the tab open. Within a minute of the last
cell starting, your GitHub Pages link shows **Open now**.

**For a talk or event:** stop any running session first, then **Save Version → Save & Run All
(Commit)**. The notebook then runs in the background, even with your laptop closed, for
`HOURS_OPEN` hours (up to 12), then closes the clinic by itself.

**Closing early:** stop the cell or the session (in the commit's page, **Cancel run**). The page
notices within a couple of minutes and shows **Closed right now**. Weigh-in cards still open.

## Good to know

- **Model size:** Kaggle's T4 GPUs have 16GB each, so the clinic accepts models up to 6GB here
  (Qwen2.5 3B fits; 7B models don't). Distillation is limited to about 1.5GB models.
- **Two GPUs, two rooms:** with `GPU T4 x2`, two visitors can be treated at the same time.
- **Kaggle's quota:** about 30 GPU hours a week. Kaggle shows your remaining hours in the right panel.
- **The public address changes** every time the notebook starts. That's why the page reads it from
  the gist: always share the GitHub Pages link, not the tunnel link.
- **Meetup wifi:** everyone on the same wifi looks like one visitor, so the notebook's daily limits
  are generous on purpose (`RUNS_PER_DAY` in the settings cell).
- **Logs:** `clinic-server.log` and `clinic-tunnel.log` in the notebook's output folder
  (`/kaggle/working`).
