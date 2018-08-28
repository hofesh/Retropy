from sklearn.decomposition import PCA

def get_ols_beta_dist(*all):
    df = get_ret_df(*all)
    n = df.shape[1]
    res = np.empty((n, n))
    for c1 in range(n):
        for c2 in range(n):
            y = df.iloc[:, c1]
            X = df.iloc[:, c2]
            beta1 = sm.OLS(y, X).fit().params[0]
            beta2 = sm.OLS(X, y).fit().params[0]
            x1 = np.array([beta1, beta2])
            x2 = np.abs(x1 - 1)
            val = x1[np.argmin(x2)]
            res[c1, c2] = val
    return pd.DataFrame(res, columns=df.columns, index=df.columns)


def get_beta_dist(*all, type):
    all = get(all)
    names = lmap(get_name, all)
    n = len(all)
    data = np.empty((n, n))
    for c1 in range(n):
        for c2 in range(n):
            if c1 == c2:
                val = 1
            else:
                y = all[c1]
                X = all[c2]
#                print(y.name, X.name)
                res = lrret(y, [X], return_res=True, show_res=False, sum1=(type=="R2"), pos_weights=(type=="R2"))
                if type == 'R2':
                    val = res['R^2']
                elif type == 'weight':
                    val = res['ser'][0]
            data[c1, c2] = val
    for c1 in range(n):
        for c2 in range(n):
            if type == "R2":
                val = max(data[c1, c2], data[c2, c1])
            elif type == "weight":
                x1 = np.array([data[c1, c2], data[c2, c1]])
                x2 = np.abs(x1 - 1)
                val = x1[np.argmin(x2)]
            data[c1, c2] = val
            data[c2, c1] = val
    df = pd.DataFrame(data, columns=names, index=names)
    return df


def get_ret_df(*lst):
    lst = get(lst, trim=True)
    df = pd.DataFrame({x.name: logret(x) for x in lst}).dropna()
    return df

def get_df(*lst):
    lst = get(lst, trim=True)
    df = pd.DataFrame({x.name: x for x in lst}).dropna()
    return df

def show_mds(*all, type='cor'):
    if type == 'cor':
        df = get_ret_df(*all)
        sim = np.corrcoef(df.T)
        dist = 1-sim
    elif type == 'cov':
#         df = get_df(*all)
        df = get_ret_df(*all)
        sim = np.cov(df.T)
        np.fill_diagonal(sim, 1)
        dist = np.abs(1-sim)
    elif type == 'weight':
        dist = get_beta_dist(*all, type='weight')
        dist = np.abs(1 - dist)
    elif type == 'R2':
        dist = get_beta_dist(*all, type='R2')
        dist = 1 - dist
    elif type == 'beta':
        dist = get_ols_beta_dist(*all)
        dist = np.abs(1 - dist)

    names = lmap(get_name, all)
    #dist = dist - dist.mean(axis=1)
    if not isinstance(dist, pd.DataFrame):
        dist = pd.DataFrame(dist, columns=names, index=names)
    display(dist)
    
    pca = PCA(n_components=2)
    tr = pca.fit_transform(dist)
    plot_scatter_xy(tr[:, 0], tr[:, 1], names=names, title=f"{type} MDS")
