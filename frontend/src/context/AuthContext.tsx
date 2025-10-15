import React, { createContext, useContext, useEffect, useMemo, useState, useCallback } from 'react';

type UserRole = 'admin' | 'advanced' | 'basic';

interface SessionInfo {
  token: string;
  username: string;
  displayName: string;
  role: UserRole;
  expiresAt: string;
}

interface AuthContextValue {
  user: SessionInfo | null;
  loading: boolean;
  login: (username: string, password: string) => Promise<void>;
  logout: () => void;
  refresh: () => Promise<void>;
}

const AuthContext = createContext<AuthContextValue | undefined>(undefined);

const STORAGE_KEY = 'firp-auth-token';

async function fetchSession(token: string): Promise<SessionInfo> {
  const resp = await fetch('/api/auth/me', {
    headers: {
      Authorization: `Bearer ${token}`,
    },
  });
  if (!resp.ok) {
    throw new Error('无法获取会话信息');
  }
  const data = await resp.json();
  return {
    token: data.token,
    username: data.username,
    displayName: data.display_name,
    role: data.role,
    expiresAt: data.expires_at,
  };
}

export const AuthProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [user, setUser] = useState<SessionInfo | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const token = localStorage.getItem(STORAGE_KEY);
    if (!token) {
      setLoading(false);
      return;
    }
    fetchSession(token)
      .then(setUser)
      .catch(() => {
        localStorage.removeItem(STORAGE_KEY);
        setUser(null);
      })
      .finally(() => setLoading(false));
  }, []);

  const login = useCallback(async (username: string, password: string) => {
    const resp = await fetch('/api/auth/login', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ username, password }),
    });
    if (!resp.ok) {
      const err = await resp.json().catch(() => ({ detail: '登录失败' }));
      throw new Error(err.detail || '登录失败');
    }
    const data = await resp.json();
    const session: SessionInfo = {
      token: data.token,
      username: data.username,
      displayName: data.display_name,
      role: data.role,
      expiresAt: data.expires_at,
    };
    localStorage.setItem(STORAGE_KEY, session.token);
    setUser(session);
  }, []);

  const logout = useCallback(() => {
    const token = localStorage.getItem(STORAGE_KEY);
    if (token) {
      fetch('/api/auth/logout', {
        method: 'POST',
        headers: { Authorization: `Bearer ${token}` },
      }).catch(() => undefined);
    }
    localStorage.removeItem(STORAGE_KEY);
    setUser(null);
  }, []);

  const refresh = useCallback(async () => {
    const token = localStorage.getItem(STORAGE_KEY);
    if (!token) {
      setUser(null);
      return;
    }
    const session = await fetchSession(token);
    setUser(session);
  }, []);

  const value = useMemo(
    () => ({ user, loading, login, logout, refresh }),
    [user, loading, login, logout, refresh],
  );

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
};

export function useAuth() {
  const ctx = useContext(AuthContext);
  if (!ctx) {
    throw new Error('useAuth 必须在 AuthProvider 中使用');
  }
  return ctx;
}

export function useAuthToken(): string | null {
  const { user } = useAuth();
  if (!user) return null;
  return user.token;
}

export type { UserRole, SessionInfo };
