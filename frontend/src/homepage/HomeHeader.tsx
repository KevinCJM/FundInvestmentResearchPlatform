import { useEffect, useRef, useState } from 'react'
import { Link } from 'react-router-dom'
import { Bars3Icon, ChevronDownIcon, MagnifyingGlassIcon, XMarkIcon } from '@heroicons/react/24/outline'
import { availableLanguages, chooseLocale, useI18n } from '../i18n/runtime'
import { moreModules, primaryModules } from './catalog'
import { HomeIcon } from './HomeIcon'

export function HomeBrand({ footer = false }: { footer?: boolean }) {
  const { s } = useI18n()
  return <Link to="/" className="home-brand" aria-label={s('app.home')}><img src="/homepage/images/brand.svg" alt="" width="36" height="36" /><span><strong>{s('app.title')}</strong><small>{footer ? 'RESEARCH WITH EVIDENCE.' : 'FUND INVESTMENT RESEARCH PLATFORM'}</small></span></Link>
}

export function HomeHeader({ onSearch, onWorkspace }: { onSearch: () => void; onWorkspace: () => void }) {
  const { s, locale } = useI18n()
  const [open, setOpen] = useState(false)
  const [mobile, setMobile] = useState(false)
  const dropdown = useRef<HTMLDivElement>(null)
  const moreButton = useRef<HTMLButtonElement>(null)
  const mobileButton = useRef<HTMLButtonElement>(null)
  useEffect(() => {
    if (!open && !mobile) return
    const keydown = (event: KeyboardEvent) => {
      if (event.key === 'Escape') { (open ? moreButton : mobileButton).current?.focus(); setOpen(false); setMobile(false) }
    }
    const outside = (event: PointerEvent) => { if (!dropdown.current?.contains(event.target as Node)) setOpen(false) }
    window.addEventListener('keydown', keydown)
    window.addEventListener('pointerdown', outside)
    return () => { window.removeEventListener('keydown', keydown); window.removeEventListener('pointerdown', outside) }
  }, [open, mobile])
  return <header className="home-header" data-testid="site-header">
    <a className="home-skip" href="#home-content">{s('landing.skip')}</a>
    <div className="home-header-inner">
      <HomeBrand />
      <nav className="home-desktop-nav" aria-label={s('navigation.main')}>
        <Link to="/" aria-current="page">{s('navigation.routes.home')}</Link>
        {primaryModules.map(item => <Link key={item.id} to={item.path}>{s(`landing.module.${item.id}`)}</Link>)}
        <div ref={dropdown} className="home-more-wrap">
          <button ref={moreButton} type="button" className="home-more-button" aria-expanded={open} aria-controls="home-more" onClick={() => setOpen(!open)}>{s('landing.more')}<ChevronDownIcon aria-hidden="true" /></button>
          {open && <div id="home-more" className="home-dropdown">{moreModules.map(item => <Link key={item.id} to={item.path} onClick={() => setOpen(false)}><HomeIcon name={item.icon} />{s(`landing.module.${item.id}`)}</Link>)}</div>}
        </div>
      </nav>
      <div className="home-header-actions">
        <button type="button" className="home-icon-button home-search-trigger" aria-label={s('landing.searchOpen')} onClick={onSearch}><MagnifyingGlassIcon aria-hidden="true" /></button>
        <label className="home-language"><HomeIcon name="globe" /><span className="sr-only">{s('i18n.language')}</span><select aria-label={s('i18n.language')} value={locale} onChange={event => void chooseLocale(event.target.value)}>{availableLanguages().filter(item => item.enabled).map(item => <option key={item.id} value={item.id}>{item.id === 'zh-CN' ? '中文' : item.label}</option>)}</select></label>
        <button type="button" className="home-avatar" aria-label={s('landing.workspaceOpen')} onClick={onWorkspace}>K</button>
        <button ref={mobileButton} type="button" className="home-icon-button home-menu-toggle" aria-label={s('navigation.menu')} aria-expanded={mobile} aria-controls="home-mobile-nav" onClick={() => setMobile(!mobile)}>{mobile ? <XMarkIcon aria-hidden="true" /> : <Bars3Icon aria-hidden="true" />}</button>
      </div>
    </div>
    {mobile && <nav id="home-mobile-nav" className="home-mobile-nav" aria-label={s('landing.mobileNav')}>
      <Link to="/" aria-current="page" onClick={() => setMobile(false)}>{s('navigation.routes.home')}</Link>
      {[...primaryModules, ...moreModules].map(item => <Link key={item.id} to={item.path} onClick={() => setMobile(false)}><HomeIcon name={item.icon} />{s(`landing.module.${item.id}`)}</Link>)}
    </nav>}
  </header>
}
